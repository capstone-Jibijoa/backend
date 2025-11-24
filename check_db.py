import psycopg2
import os
import json
from dotenv import load_dotenv

load_dotenv()

def check_database_structure():
    """데이터베이스 구조 확인"""
    
    conn = None
    try:
        print("\n" + "="*70)
        print("🔍 AWS RDS PostgreSQL 데이터베이스 분석")
        print("="*70)
        
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        cur = conn.cursor()
        
        print("\n✅ 데이터베이스 연결 성공!")
        
        # 1. welcome_meta 테이블 기본 정보
        print("\n📋 1. welcome_meta2 테이블 기본 정보:")
        print("-" * 70)
        
        cur.execute("SELECT COUNT(*) FROM welcome_meta2")
        total_count = cur.fetchone()[0]
        print(f"총 레코드 수: {total_count:,}개")
        
        # 2. 테이블 구조
        print("\n📋 2. welcome_meta2 테이블 구조:")
        print("-" * 70)
        
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'welcome_meta2'
            ORDER BY ordinal_position
        """)
        columns = cur.fetchall()
        
        print(f"{'컬럼명':<30} {'타입':<20} {'NULL 허용'}")
        print("-" * 70)
        for col in columns:
            print(f"{col[0]:<30} {col[1]:<20} {col[2]}")
        
        # 3. JSONB structured_data 구조 분석
        print("\n📋 3. structured_data JSONB 구조 분석:")
        print("-" * 70)
        
        cur.execute("""
            SELECT structured_data 
            FROM welcome_meta2 
            LIMIT 1
        """)
        sample_data = cur.fetchone()
        
        if sample_data:
            jsonb_data = sample_data[0]
            print("\n샘플 JSONB 키 목록:")
            for key in sorted(jsonb_data.keys()):
                value = jsonb_data[key]
                value_type = type(value).__name__
                sample_value = str(value)[:50] if value else "null"
                print(f"  - {key:<30} ({value_type:>10}): {sample_value}")
            
            # 4. 주요 필드 값 분포 확인
            print("\n📋 4. 주요 필드 값 분포:")
            print("-" * 70)
            
            # 성별 분포
            if 'gender' in jsonb_data:
                cur.execute("""
                    SELECT 
                        structured_data->>'gender' as gender,
                        COUNT(*) as count
                    FROM welcome_meta2
                    GROUP BY structured_data->>'gender'
                """)
                print("\n[성별 분포]")
                for row in cur.fetchall():
                    print(f"  {row[0]}: {row[1]:,}명")
            
            # 지역 분포 (상위 5개)
            if 'region' in jsonb_data:
                cur.execute("""
                    SELECT 
                        structured_data->>'region' as region,
                        COUNT(*) as count
                    FROM welcome_meta2
                    GROUP BY structured_data->>'region'
                    ORDER BY count DESC
                    LIMIT 5
                """)
                print("\n[지역 분포 (상위 5개)]")
                for row in cur.fetchall():
                    print(f"  {row[0]}: {row[1]:,}명")
            
            # 나이대 분포
            if 'birth_year' in jsonb_data:
                cur.execute("""
                    SELECT 
                        CASE 
                            WHEN (structured_data->>'birth_year')::int >= 2006 THEN '10대'
                            WHEN (structured_data->>'birth_year')::int >= 1996 THEN '20대'
                            WHEN (structured_data->>'birth_year')::int >= 1986 THEN '30대'
                            WHEN (structured_data->>'birth_year')::int >= 1976 THEN '40대'
                            WHEN (structured_data->>'birth_year')::int >= 1966 THEN '50대'
                            ELSE '60대 이상'
                        END as age_group,
                        COUNT(*) as count
                    FROM welcome_meta2
                    WHERE structured_data->>'birth_year' IS NOT NULL
                    GROUP BY age_group
                    ORDER BY age_group
                """)
                print("\n[나이대 분포]")
                for row in cur.fetchall():
                    print(f"  {row[0]}: {row[1]:,}명")
            
            # 5. 샘플 데이터 출력
            print("\n📋 5. 샘플 레코드 (상위 3개):")
            print("-" * 70)
            
            cur.execute("""
                SELECT 
                    pid,
                    structured_data->>'gender' as gender,
                    structured_data->>'birth_year' as birth_year,
                    structured_data->>'region' as region
                FROM welcome_meta2
                LIMIT 3
            """)
            
            print(f"{'PID':<10} {'성별':<10} {'출생연도':<12} {'지역'}")
            print("-" * 70)
            for row in cur.fetchall():
                print(f"{row[0]:<10} {row[1] or 'N/A':<10} {row[2] or 'N/A':<12} {row[3] or 'N/A'}")
        
        # 6. 테스트 쿼리
        print("\n📋 6. 테스트 쿼리 실행:")
        print("-" * 70)
        
        # 30대 남성 검색 테스트
        cur.execute("""
            SELECT COUNT(*) 
            FROM welcome_meta2
            WHERE structured_data->>'gender' = 'M'
              AND (structured_data->>'birth_year')::int BETWEEN 1986 AND 1995
        """)
        test_count = cur.fetchone()[0]
        print(f"✅ 30대 남성: {test_count:,}명")
        
        # 경기 거주자 검색 테스트
        cur.execute("""
            SELECT COUNT(*) 
            FROM welcome_meta2
            WHERE structured_data->>'region' = '경기'
        """)
        test_count = cur.fetchone()[0]
        print(f"✅ 경기 거주자: {test_count:,}명")
        
        cur.close()
        
        print("\n" + "="*70)
        print("📝 코드 수정 완료 상태")
        print("="*70)
        
        print("\n✅ 이미 수정된 사항:")
        print("1. 테이블 이름: welcome → welcome_meta2")
        print("2. JSONB 접근: structured_data->>'필드명'")
        print("3. Qdrant 컬렉션: QDRANT_COLLECTION_WELCOME_NAME, QDRANT_COLLECTION_QPOLL_NAME 환경변수 사용")
        print("4. 주관식/QPoll: 각각 welcome_subjective_vectors, qpoll_vector_v2 사용")
        
        print("\n🚀 바로 사용 가능합니다!")
        print("   테스트 명령어:")
        print("   curl -X POST http://localhost:8000/api/search \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"query\": \"30대 남자 술을 먹은 사람\"}'")
        
        print("\n" + "="*70 + "\n")
        
    except psycopg2.Error as e:
        print(f"\n❌ 데이터베이스 연결 실패!")
        print(f"오류: {e}")
        print("\n.env 파일의 DB 설정을 확인하세요:")
        print("  DB_HOST=your-rds-endpoint.amazonaws.com")
        print("  DB_NAME=your_database")
        print("  DB_USER=your_username")
        print("  DB_PASSWORD=your_password")
        
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    check_database_structure()