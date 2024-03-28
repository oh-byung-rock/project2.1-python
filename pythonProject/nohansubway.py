import requests
import pandas as pd

base_url = "https://apis.openapi.sk.com/puzzle/subway/congestion/stat/car/stations/213"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "appKey": ""
}

time_range = "18"
daynames = ["MON", "TUE", "WED", "THU", "FRI"]

# 초기화
congestion_data = []

# congestion 관련
for dayname in daynames:
    response = requests.get(base_url, headers=headers, params={"hh": time_range, "dow": dayname})
    data = response.json()

    for stat in data["contents"]["stat"]:
        updn_line = stat["updnLine"]

        for congestion in stat["data"]:
            dow = congestion["dow"]
            hh = congestion["hh"]
            mm = congestion["mm"]

            congestion_car = congestion["congestionCar"]
            congestion_car1 = congestion_car[0]
            congestion_car2 = congestion_car[1]
            congestion_car3 = congestion_car[2]
            congestion_car4 = congestion_car[3]
            congestion_car5 = congestion_car[4]
            congestion_car6 = congestion_car[5]
            congestion_car7 = congestion_car[6]
            congestion_car8 = congestion_car[7]
            congestion_car9 = congestion_car[8]
            congestion_car10 = congestion_car[9]

            # 데이터 추가
            congestion_data.append({
                "updnLine": updn_line,
                "dow": dow,
                "hh": hh,
                "mm": mm,
                "congestionCar1": congestion_car1,
                "congestionCar2": congestion_car2,
                "congestionCar3": congestion_car3,
                "congestionCar4": congestion_car4,
                "congestionCar5": congestion_car5,
                "congestionCar6": congestion_car6,
                "congestionCar7": congestion_car7,
                "congestionCar8": congestion_car8,
                "congestionCar9": congestion_car9,
                "congestionCar10": congestion_car10
            })

# csv로 저장
df = pd.DataFrame(congestion_data)
df.to_csv("nohan_congest.csv", index=False, encoding="utf-8")
