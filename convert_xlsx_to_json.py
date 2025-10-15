import pandas as pd
import json
import numpy as np

# --- 0. 설정 및 최종 컬럼 매핑 정의 ---

FILE_PATHS = {
    'file1': 'data/welcome1.xlsx',
    'file2': 'data/welcome2.xlsx'
}
MERGE_KEY = 'panel_id'
OUTPUT_JSON_PATH = 'merged_data.json'

# 최종 영문 키 매핑 (Question Text -> English Key)
FINAL_COLUMN_MAPPING = {
    # --- ID Fields --- String
    'x': 'panel_id',
    'mb_sn': 'panel_id',
    '패널ID': 'panel_id',

    # --- welcome1 ---
    '귀하의 성별은': 'gender',
    '귀하의 출생년도는 어떻게 되십니까?': 'birth_year',
    '회원님께서 현재 살고 계신 지역은 어디인가요?': 'region_major',
    '그렇다면, [VALUE:Q12]의 어느 구에 살고 계신가요?': 'region_minor',

    # --- welcome2 ---
    '결혼여부': 'marital_status',
    '자녀수': 'children_count', # numeric
    '가족수': 'family_size',
    '최종학력': 'education_level',
    '직업': 'job_title_raw',
    '직무': 'job_duty',
    '월평균 개인소득': 'income_personal_monthly',
    '월평균 가구소득': 'income_household_monthly',
    '보유전제품': 'owned_electronics',  # multi
    '보유 휴대폰 단말기 브랜드': 'phone_brand',
    '보유 휴대폰 모델명': 'phone_model_raw',
    '보유차량여부': 'car_ownership',
    '자동차 제조사': 'car_manufacturer',
    '자동차 모델': 'car_model_raw',
    '흡연경험': 'smoking_experience',   # multi
    '흡연경험 담배브랜드': 'smoking_brand', # multi
    '흡연경험 담배브랜드(기타브랜드)': 'smoking_brand_etc',
    '궐련형 전자담배/가열식 전자담배 이용경험': 'e_cigarette_experience',   # multi
    '흡연경험 담배 브랜드(기타내용)': 'smoking_brand_other_details',    # multi
    '음용경험 술': 'drinking_experience',
    '음용경험 술(기타내용)': 'drinking_experience_other_details',   # string
}

# --- 1. 파일 로드 및 3단계 컬럼명 변환 함수 ---

def load_and_standardize_file(path, final_mapping):
    """
    Parses a non-standard label sheet to map values and standardize column names.
    Returns the processed DataFrame and the generated label maps.
    """
    try:
        xlsx = pd.ExcelFile(path)
        df_label = xlsx.parse(xlsx.sheet_names[1])

        qcode_to_question = {}
        value_labels = {}
        current_q_code = None

        # 레이블 시트의 모든 행을 순차적으로 반복
        for row in df_label.itertuples(index=False, name=None):
            col_a, col_b = row[0], row[1]

            if pd.isna(col_a) and pd.isna(col_b):
                current_q_code = None
                continue

            is_q_code = isinstance(col_a, str) and not col_a.isnumeric()

            # 질문 코드(Q-code)와 문항 텍스트('직업')가 모두 존재하는 행을 찾는다
            if is_q_code and pd.notna(col_b):
                current_q_code = col_a
                # 컬럼명 변경용) Q-code를 key로, 문항 텍스트를 값으로 하는 dictionary(qcode_to_question) 생성
                qcode_to_question[current_q_code] = col_b
                # 해당 Q-code 아래에 value-label mapping 을 저장할 새 딕셔너리 준비
                value_labels[current_q_code] = {}
            # 질문 코드 정의가 된 상태(current_q_code != None)에서 하위행에 있는 숫자 코드(col_a)와 응답 텍스트(col_b) 쌍을 찾는다
            elif current_q_code and pd.notna(col_a) and pd.notna(col_b):
                # col_a를 숫자로 변환하여 응답 매핑 사전(value_labels)생성한다
                try:
                    value = float(col_a)
                    if value.is_integer():
                        value = int(value)
                    value_labels[current_q_code][value] = col_b
                except (ValueError, TypeError):
                    continue
        
        # data sheet
        df_data = xlsx.parse(xlsx.sheet_names[0])

        # 데이터 시트의 모든 컬럼 순회하면서
        for col in df_data.columns:
            # 해당 컬럼(Q-code)에 대해 value_labels가 존재하는지 확인한다
            if col in value_labels:
                # 데이터 시트 내에서 해당 컬럼의 숫자 값을 준비된 한글 응답 텍스트로 일관 치환한다
                df_data[col] = df_data[col].replace(value_labels[col])

        # Q-code 를 문항 텍스트로 변경한 후, final_mapping을 사용해 영문키로 변경 -> 컬럼 매핑 딕셔너리 생성
        renamer_dict = {}
        for current_col_name in df_data.columns:
            question_text = qcode_to_question.get(current_col_name, current_col_name)
            final_key = final_mapping.get(question_text, final_mapping.get(current_col_name, current_col_name))
            renamer_dict[current_col_name] = final_key

        # 생성된 딕셔너리를 사용해 컬럼명을 최종 적용
        df_data.rename(columns=renamer_dict, inplace=True)
        
        # 결측치(NaN)을 None으로 변환하고, 처리된 데이터프레임과 함께 생성된 두 매핑 사전 반환
        return df_data.replace({np.nan: None}), qcode_to_question, value_labels

    except Exception as e:
        print(f"파일 처리 중 오류 발생: {path}, 오류: {e}")
        raise

# --- 2. 통합 및 JSON 변환 ---

def integrate_and_finalize(file_paths, final_mapping):
    """Integrates files, handles multi-select fields, and finalizes the DataFrame."""
    
    df1, q_map1, v_map1 = load_and_standardize_file(file_paths['file1'], final_mapping)
    df2, q_map2, v_map2 = load_and_standardize_file(file_paths['file2'], final_mapping)
    
    # Merge DataFrames
    df_merged = pd.merge(df1, df2, on=MERGE_KEY, how='outer', suffixes=('_f1', '_f2'))

    # Merge label maps
    q_map_all = {**q_map1, **q_map2}
    v_map_all = {**v_map1, **v_map2}

    # --- Handle Multi-Select Fields Post-Merge ---
    multi_select_keys = [
        'owned_electronics', 'smoking_experience', 'smoking_brand',
        'e_cigarette_experience', 'drinking_experience', 'smoking_brand_other_details'
    ]

    # Create reverse map to find Q-code from English key
    inverted_final_map = {v: k for k, v in final_mapping.items()}
    inverted_q_map = {v: k for k, v in q_map_all.items()}

    def apply_multi_select_labels(value_str, labels):
        if not isinstance(value_str, str):
            return value_str # Return as is if not a string
        
        labeled_values = []
        # Split by comma and handle potential spaces
        for val in value_str.split(','):
            val = val.strip()
            if not val:
                continue
            try:
                num_val = int(val)
                labeled_values.append(labels.get(num_val, num_val))
            except (ValueError, TypeError):
                labeled_values.append(val) # Keep as is if not a number
        return labeled_values

    for key in multi_select_keys:
        if key in df_merged.columns:
            question_text = inverted_final_map.get(key)
            q_code = inverted_q_map.get(question_text)
            
            if q_code and q_code in v_map_all:
                labels = v_map_all[q_code]
                df_merged[key] = df_merged[key].apply(lambda x: apply_multi_select_labels(x, labels))

    return df_merged

# --- 메인 실행 ---
if __name__ == '__main__':
    try:
        final_df = integrate_and_finalize(FILE_PATHS, FINAL_COLUMN_MAPPING)
        final_json_list = final_df.to_dict('records')
        
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_json_list, f, ensure_ascii=False, indent=4)
            
        print(f"{len(FILE_PATHS)}개 파일 통합 및 최종 JSON 변환 완료.")
        print(f"총 통합 레코드 수: {len(final_json_list)}")
        print(f"결과가 '{OUTPUT_JSON_PATH}' 파일에 저장되었습니다.")
        
        if final_json_list:
            print("\n--- 첫 번째 고객의 통합 JSON 구조 (예시) ---")
            print(json.dumps(final_json_list[0], indent=4, ensure_ascii=True))
            
    except Exception as e:
        print(f"\n--- 최종 프로세스 오류 ---")
        print(f"프로세스 실행 실패: {e}")
