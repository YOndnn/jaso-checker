"""판교_8반_이진호_파이프라인실습

[Day 1 종합실습] 비동기 데이터 수집 미니 파이프라인
- 하는 일:
    1) fetch_json()   : httpx.AsyncClient로 API 하나를 안전하게 호출 (예외 처리)
    2) collect_all()  : asyncio.gather()로 3개 API(Open-Meteo, countries.dev, ip-api) 동시 수집
    3) CombinedRecord : Pydantic v2 스키마로 타입·범위 검증 (성공/실패 분리)
    4) 저장 + 성능 비교 : 검증 통과 데이터를 CSV/Parquet 두 형식으로 저장, 쓰기/읽기 시간 비교
- 사용 API:
    - Open-Meteo  : 서울 3일 시간대별 기온·강수확률
    - countries.dev : 한국 국가 정보
    - ip-api      : IP(8.8.8.8) 기반 지역 정보
- 변경 내역:
    2026-08-06  최초 작성 (이진호)
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent

WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=37.5665&longitude=126.9780"
    "&hourly=temperature_2m,precipitation_probability"
    "&forecast_days=3&timezone=Asia/Seoul"
)
COUNTRY_URL = "https://countries.dev/alpha/KR"
IP_URL = "http://ip-api.com/json/8.8.8.8"


# 1) 비동기 수집 -------------------------------------------------------------
async def fetch_json(client: httpx.AsyncClient, name: str, url: str) -> dict | None:
    """API 하나를 호출해 JSON을 돌려준다. 실패하면 None + logger.error."""
    try:
        resp = await client.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"{name} 수집 성공")
        return data
    except (httpx.HTTPError, ValueError) as e:
        logger.error(f"{name} 수집 실패: {e}")
        return None


async def collect_all() -> dict[str, dict | None]:
    """3개 API를 asyncio.gather()로 동시에 호출한다."""
    async with httpx.AsyncClient() as client:
        weather, country, ip = await asyncio.gather(
            fetch_json(client, "open-meteo", WEATHER_URL),
            fetch_json(client, "countries.dev", COUNTRY_URL),
            fetch_json(client, "ip-api", IP_URL),
        )
    return {"weather": weather, "country": country, "ip": ip}


# 2) Pydantic v2 스키마 -------------------------------------------------------
class CombinedRecord(BaseModel):
    """weather(시간대별 1건) + country + ip 정보를 한 행으로 합친 검증 스키마."""

    time: datetime
    temperature_c: float = Field(ge=-90, le=60, description="기온, 현실적인 범위(-90~60도)")
    precipitation_probability: int = Field(ge=0, le=100, description="강수확률 %")
    country_name: str = Field(min_length=1)
    capital: str = Field(min_length=1)
    population: int = Field(gt=0)
    region: str = Field(min_length=1)
    ip_query: str = Field(min_length=1)
    ip_country: str = Field(min_length=1)
    ip_city: str = Field(min_length=1)
    isp: str = Field(min_length=1)


# 3) 원본 응답 → 행(rows) 추출 -------------------------------------------------
def build_rows(weather: dict, country: dict, ip: dict) -> list[dict[str, Any]]:
    """3개 API 응답에서 필요한 필드만 추출해 시간대별 행 리스트로 합친다."""
    hourly = weather["hourly"]
    common = {
        "country_name": country.get("name"),
        "capital": country.get("capital"),
        "population": country.get("population"),
        "region": country.get("region"),
        "ip_query": ip.get("query"),
        "ip_country": ip.get("country"),
        "ip_city": ip.get("city"),
        "isp": ip.get("isp"),
    }
    rows = []
    for t, temp, precip in zip(
        hourly["time"], hourly["temperature_2m"], hourly["precipitation_probability"]
    ):
        rows.append(
            {"time": t, "temperature_c": temp, "precipitation_probability": precip, **common}
        )
    return rows


# 4) 검증 파이프라인 (valid / errors 분리) --------------------------------------
def validate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """rows를 순회하며 CombinedRecord로 변환한다.

    성공 → valid(dict 리스트), 실패 → errors({row, error}) 리스트.
    타입/범위가 어긋난 값은 ValidationError로 잡아 errors에 분리한다.
    """
    valid: list[dict] = []
    errors: list[dict] = []
    for i, row in enumerate(rows):
        try:
            record = CombinedRecord(**row)
            valid.append(record.model_dump())
        except ValidationError as e:
            errors.append({"row": i, "error": str(e)})
    return valid, errors


# 5) 저장 + 성능 비교 ---------------------------------------------------------
def save_and_compare(valid: list[dict]) -> None:
    """valid를 CSV/Parquet 두 형식으로 저장하고 쓰기·읽기 시간·용량을 비교 출력한다."""
    df = pd.DataFrame(valid)
    csv_path = HERE / "weather_valid.csv"
    parquet_path = HERE / "weather_valid.parquet"

    t0 = time.perf_counter()
    df.to_csv(csv_path, index=False)
    csv_write = time.perf_counter() - t0

    t0 = time.perf_counter()
    df.to_parquet(parquet_path, index=False)
    parquet_write = time.perf_counter() - t0

    t0 = time.perf_counter()
    pd.read_csv(csv_path)
    csv_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    pd.read_parquet(parquet_path)
    parquet_read = time.perf_counter() - t0

    print(f"{len(valid)}건 저장 완료 → {csv_path.name}, {parquet_path.name}")
    print(
        f"  CSV     : 쓰기 {csv_write * 1000:.2f}ms / 읽기 {csv_read * 1000:.2f}ms "
        f"/ 크기 {csv_path.stat().st_size}B"
    )
    print(
        f"  Parquet : 쓰기 {parquet_write * 1000:.2f}ms / 읽기 {parquet_read * 1000:.2f}ms "
        f"/ 크기 {parquet_path.stat().st_size}B"
    )


async def main() -> None:
    results = await collect_all()
    assert all(results.values()), "API 3개 중 일부가 실패했다"  # 3개 모두 응답 정상 확인
    print("API 3개 동시 수집 완료 (asyncio.gather)")
    print()

    rows = build_rows(results["weather"], results["country"], results["ip"])

    # 검증 실패 경로를 시연하기 위해 일부러 값이 틀린 행 하나를 추가한다
    # (population 음수, isp 빈 문자열 → ValidationError 2건 발생 예상)
    bad_row = {**rows[0], "population": -1, "isp": ""}
    rows.append(bad_row)

    valid, errors = validate(rows)
    print(f"검증 결과 → 유효 {len(valid)}건 / 오류 {len(errors)}건")
    for err in errors:
        first_line = err["error"].splitlines()[0]
        print(f"  - row {err['row']}: {first_line}")
    assert len(errors) == 1  # 일부러 넣은 오류 행 1건만 걸러져야 한다
    print()

    save_and_compare(valid)


if __name__ == "__main__":
    asyncio.run(main())
