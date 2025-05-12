from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
from datetime import datetime, timedelta
from PIL import Image
import pytz

# 저장 경로 설정
SAVE_DIR = r"D:\drl-candlesticks-trader-main1\paper1\data\Chart Data\1d"
os.makedirs(SAVE_DIR, exist_ok=True)

# 시작일과 종료일 설정
START_DATE = datetime(2021, 10, 12)
END_DATE = datetime(2023, 12, 19)

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def capture_chart(driver, date):
    try:
        # 바이낸스 차트 URL
        url = f"https://www.binance.com/en/trade/BTC_USDT?type=spot&interval=1d&startTime={int(date.timestamp() * 1000)}"
        driver.get(url)
        time.sleep(5)  # 차트 로딩 대기
        
        # 차트 영역 찾기
        chart_element = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.chart-container"))
        )
        
        # 스크린샷 저장
        filename = f"BTC_USDT_1d_{date.strftime('%Y%m%d')}.png"
        filepath = os.path.join(SAVE_DIR, filename)
        chart_element.screenshot(filepath)
        
        print(f"캡처 완료: {filename}")
        return True
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return False

def main():
    driver = setup_driver()
    current_date = START_DATE
    
    try:
        while current_date <= END_DATE:
            if capture_chart(driver, current_date):
                print(f"{current_date.strftime('%Y-%m-%d')} 차트 캡처 성공")
            else:
                print(f"{current_date.strftime('%Y-%m-%d')} 차트 캡처 실패")
            
            current_date += timedelta(minutes=15)  # 15분 간격으로 증가
            time.sleep(2)  # 요청 간격 조절
            
    finally:
        driver.quit()
        print("모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main() 