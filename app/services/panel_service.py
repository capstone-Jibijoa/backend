import asyncio
import logging
from typing import Dict
from app.repositories.panel_repo import PanelRepository
from app.repositories.qpoll_repo import QpollRepository

class PanelService:
    def __init__(self):
        self.panel_repo = PanelRepository()
        self.qpoll_repo = QpollRepository()

    async def get_panel_details(self, panel_id: str) -> Dict:
        """패널 상세 정보 조회 (DB + Qdrant 병합)"""
        try:
            # 병렬 조회
            results = await asyncio.gather(
                asyncio.to_thread(self.panel_repo.get_panel_detail, panel_id),
                asyncio.to_thread(self.qpoll_repo.get_all_responses_by_panel, panel_id),
                return_exceptions=True
            )
            
            panel_data = results[0] if not isinstance(results[0], Exception) and results[0] else {}
            qpoll_data = results[1] if not isinstance(results[1], Exception) and results[1] else {}

            # 데이터 병합
            merged_data = {**panel_data, **qpoll_data}
            
            if not merged_data:
                raise ValueError(f"Panel not found: {panel_id}")
                
            return merged_data
            
        except Exception as e:
            logging.error(f"패널 상세 조회 실패: {e}")
            raise e