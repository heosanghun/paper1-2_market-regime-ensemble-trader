import os
import time
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingViewCapturer:
    def __init__(self):
        load_dotenv()
        self.username = os.getenv('TRADINGVIEW_USERNAME')
        self.password = os.getenv('TRADINGVIEW_PASSWORD')
        self.base_url = "https://www.tradingview.com/chart/"
        self.setup_driver()
        
    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        
    def login(self):
        try:
            self.driver.get("https://www.tradingview.com/accounts/signin/")
            time.sleep(3)
            
            # 로그인 폼 입력
            username_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "username"))
            )
            password_field = self.driver.find_element(By.NAME, "password")
            
            username_field.send_keys(self.username)
            password_field.send_keys(self.password)
            
            # 로그인 버튼 클릭
            login_button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            login_button.click()
            
            time.sleep(5)
            logger.info("트레이딩뷰 로그인 성공")
            
        except Exception as e:
            logger.error(f"로그인 실패: {str(e)}")
            raise
            
    def set_timeframe(self, interval):
        try:
            # 시간봉 버튼 클릭
            timeframe_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-name='timeframe-button']"))
            )
            timeframe_button.click()
            
            # 시간봉 선택
            if interval == "1d":
                timeframe = self.driver.find_element(By.XPATH, "//div[text()='1D']")
            elif interval == "1w":
                timeframe = self.driver.find_element(By.XPATH, "//div[text()='1W']")
                
            timeframe.click()
            time.sleep(2)
            logger.info(f"{interval} 시간봉 설정 완료")
            
        except Exception as e:
            logger.error(f"시간봉 설정 실패: {str(e)}")
            raise
            
    def set_date_range(self, start_date, end_date):
        try:
            # 커스텀 레인지 버튼 클릭
            range_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-name='date-range-button']"))
            )
            range_button.click()
            
            # 커스텀 레인지 선택
            custom_range = self.driver.find_element(By.XPATH, "//div[text()='Custom Range']")
            custom_range.click()
            
            # 날짜 입력
            start_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[data-name='start-date']"))
            )
            end_input = self.driver.find_element(By.CSS_SELECTOR, "input[data-name='end-date']")
            
            start_input.clear()
            end_input.clear()
            
            start_input.send_keys(start_date.strftime("%Y-%m-%d"))
            end_input.send_keys(end_date.strftime("%Y-%m-%d"))
            
            # 적용 버튼 클릭
            apply_button = self.driver.find_element(By.XPATH, "//button[text()='Apply']")
            apply_button.click()
            
            time.sleep(3)
            logger.info(f"날짜 범위 설정 완료: {start_date} ~ {end_date}")
            
        except Exception as e:
            logger.error(f"날짜 범위 설정 실패: {str(e)}")
            raise
            
    def capture_chart(self, interval, start_date, end_date, save_path):
        try:
            # 차트 URL 설정
            chart_url = f"{self.base_url}?symbol=BINANCE:BTCUSDT&interval={interval}"
            self.driver.get(chart_url)
            time.sleep(5)
            
            # 시간봉 설정
            self.set_timeframe(interval)
            
            # 날짜 범위 설정
            self.set_date_range(start_date, end_date)
            
            # 차트가 로드될 때까지 대기
            time.sleep(5)
            
            # 스크린샷 저장
            os.makedirs(save_path, exist_ok=True)
            filename = f"BTCUSDT_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_tradingview.png"
            filepath = os.path.join(save_path, filename)
            
            self.driver.save_screenshot(filepath)
            logger.info(f"차트 캡처 완료: {filepath}")
            
        except Exception as e:
            logger.error(f"차트 캡처 실패: {str(e)}")
            raise
            
    def close(self):
        self.driver.quit()
        
def capture_all_charts():
    capturer = TradingViewCapturer()
    try:
        # 로그인
        capturer.login()
        
        # 시간봉별 설정
        intervals = {
            "1d": 30,  # 30일 단위
            "1w": 180  # 180일 단위
        }
        
        # 저장 경로
        base_dir = "./captures"
        
        # 시작일과 종료일 설정
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)  # 5년치
        
        # 주요 이벤트 구간
        event_periods = [
            ('2022-11-01', '2022-11-30'),  # FTX
            ('2022-05-01', '2022-05-31'),  # 테라/루나
            ('2020-03-01', '2020-03-31'),  # 코로나
            ('2020-05-01', '2020-05-31'),  # 반감기
        ]
        
        # 각 시간봉별로 차트 캡처
        for interval, window_days in intervals.items():
            save_path = os.path.join(base_dir, interval)
            os.makedirs(save_path, exist_ok=True)
            
            # 슬라이딩 윈도우 방식으로 캡처
            current_date = start_date
            while current_date + timedelta(days=window_days) <= end_date:
                window_end = current_date + timedelta(days=window_days)
                capturer.capture_chart(interval, current_date, window_end, save_path)
                current_date += timedelta(days=window_days)
            
            # 주요 이벤트 구간 캡처
            for ev_start, ev_end in event_periods:
                ev_start_date = datetime.strptime(ev_start, "%Y-%m-%d")
                ev_end_date = datetime.strptime(ev_end, "%Y-%m-%d")
                capturer.capture_chart(interval, ev_start_date, ev_end_date, save_path)
                
    except Exception as e:
        logger.error(f"차트 캡처 중 오류 발생: {str(e)}")
    finally:
        capturer.close()

if __name__ == "__main__":
    capture_all_charts() 