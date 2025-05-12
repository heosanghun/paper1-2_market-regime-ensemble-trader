from important_periods import get_important_periods
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
from dotenv import load_dotenv
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta

# .env에서 계정 정보 불러오기
load_dotenv("D:/drl-candlesticks-trader-main1/.env")
TRADINGVIEW_USERNAME = os.getenv("TRADINGVIEW_USERNAME")
TRADINGVIEW_PASSWORD = os.getenv("TRADINGVIEW_PASSWORD")

# 트레이딩뷰 심볼 및 시간봉 매핑
symbol = "BINANCE:BTCUSDT"
intervals = {
    "5m": "5",
    "15m": "15",
    "1h": "60",
    "4h": "240",
    "1d": "D",
    "1w": "W"
}

# 캡처 저장 경로
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "captures"))
os.makedirs(base_dir, exist_ok=True)
print(f"캡처 이미지가 다음 경로에 저장됩니다: {base_dir}")

# 각 시간봉별 폴더 생성
timeframe_dirs = {}
for name in intervals.keys():
    dir_path = os.path.join(base_dir, name)
    os.makedirs(dir_path, exist_ok=True)
    timeframe_dirs[name] = dir_path
    print(f"{name} 차트 이미지 저장 경로: {dir_path}")

# 중요 구간 가져오기
important_dates = get_important_periods()
print(f"총 {len(important_dates)}개의 중요 구간에 대해 캡처를 진행합니다.")

# 크롬 옵션 설정
options = Options()
# options.add_argument("--headless")  # 헤드리스 모드 비활성화
options.add_argument("--window-size=1920,1080")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--disable-extensions")
options.add_argument("--disable-infobars")
options.add_argument("--disable-notifications")
options.add_argument("--disable-popup-blocking")
options.add_argument("--disable-web-security")
options.add_argument("--allow-running-insecure-content")
options.add_argument("--ignore-certificate-errors")
options.add_argument("--log-level=3")
options.add_argument("--silent")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

def tradingview_login():
    try:
        driver.get("https://www.tradingview.com/chart/")
        time.sleep(15)  # 페이지 로딩 대기
        
        # 로그인 버튼 클릭
        login_btn = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'tv-header__user-menu-button')]"))
        )
        login_btn.click()
        time.sleep(2)
        
        # 이메일 입력
        email_input = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.NAME, "username"))
        )
        email_input.send_keys(TRADINGVIEW_USERNAME)
        
        # 비밀번호 입력
        pw_input = driver.find_element(By.NAME, "password")
        pw_input.send_keys(TRADINGVIEW_PASSWORD)
        
        # 로그인 버튼 클릭
        submit_btn = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))
        )
        submit_btn.click()
        time.sleep(10)
        
        print("트레이딩뷰 로그인 성공!")
        return True
    except Exception as e:
        print(f"로그인 중 오류 발생: {str(e)}")
        return False

def capture_chart(interval, date):
    try:
        # 차트 URL로 이동
        chart_url = f"https://www.tradingview.com/chart/?symbol={symbol}"
        driver.get(chart_url)
        time.sleep(10)
        
        # 시간봉 설정
        timeframe_btn = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'interval-dialog-button')]"))
        )
        timeframe_btn.click()
        time.sleep(2)
        
        # 시간봉 선택
        interval_btn = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, f"//div[contains(@class, 'item') and contains(text(), '{interval}')]"))
        )
        interval_btn.click()
        time.sleep(5)
        
        # 날짜 선택
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        end_date = date_obj + timedelta(days=1)  # 하루 단위로 캡처
        
        # 날짜 범위 버튼 클릭
        date_range_btn = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'date-range-button')]"))
        )
        date_range_btn.click()
        time.sleep(2)
        
        # 시작일 입력
        start_input = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//input[@name='start']"))
        )
        start_input.clear()
        start_input.send_keys(Keys.CONTROL + "a")
        start_input.send_keys(Keys.DELETE)
        start_input.send_keys(date)
        time.sleep(1)
        
        # 종료일 입력
        end_input = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//input[@name='end']"))
        )
        end_input.clear()
        end_input.send_keys(Keys.CONTROL + "a")
        end_input.send_keys(Keys.DELETE)
        end_input.send_keys(end_date.strftime("%Y-%m-%d"))
        time.sleep(1)
        
        # 적용 버튼 클릭
        apply_btn = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'apply-button')]"))
        )
        apply_btn.click()
        time.sleep(10)
        
        # 전체화면 캡처
        screenshot_path = os.path.join(timeframe_dirs[interval], f"BTCUSDT_{interval}_{date}_tradingview.png")
        driver.save_screenshot(screenshot_path)
        print(f"{interval} {date} 차트 캡처 완료!")
        return True
    except Exception as e:
        print(f"차트 캡처 중 오류 발생: {str(e)}")
        return False

try:
    if tradingview_login():
        for date in important_dates:
            for name, interval in intervals.items():
                if capture_chart(interval, date):
                    print(f"{name} {date} 캡처 성공!")
                else:
                    print(f"{name} {date} 캡처 실패!")
                time.sleep(5)  # 요청 간격 조절
finally:
    driver.quit()
    print("모든 작업 완료!") 