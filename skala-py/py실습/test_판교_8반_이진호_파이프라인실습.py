"""판교_8반_이진호_파이프라인실습 스키마 검증 테스트 (네트워크 호출 없이 순수 로직만 확인)."""

import pytest
from pydantic import ValidationError

from 판교_8반_이진호_파이프라인실습 import CombinedRecord, build_rows, validate

GOOD_ROW = {
    "time": "2026-08-06T00:00",
    "temperature_c": 28.5,
    "precipitation_probability": 20,
    "country_name": "Korea (Republic of)",
    "capital": "Seoul",
    "population": 51780579,
    "region": "Asia",
    "ip_query": "8.8.8.8",
    "ip_country": "United States",
    "ip_city": "Ashburn",
    "isp": "Google LLC",
}


def test_valid_row_passes():
    record = CombinedRecord(**GOOD_ROW)
    assert record.capital == "Seoul"


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("population", -1),  # gt=0 위반
        ("precipitation_probability", 150),  # le=100 위반
        ("isp", ""),  # min_length=1 위반
    ],
)
def test_invalid_row_raises(field, bad_value):
    bad_row = {**GOOD_ROW, field: bad_value}
    with pytest.raises(ValidationError):
        CombinedRecord(**bad_row)


def test_build_rows_merges_three_sources():
    weather = {
        "hourly": {
            "time": ["2026-08-06T00:00", "2026-08-06T01:00"],
            "temperature_2m": [28.5, 27.9],
            "precipitation_probability": [10, 20],
        }
    }
    country = {"name": "Korea (Republic of)", "capital": "Seoul", "population": 51780579, "region": "Asia"}
    ip = {"query": "8.8.8.8", "country": "United States", "city": "Ashburn", "isp": "Google LLC"}

    rows = build_rows(weather, country, ip)

    assert len(rows) == 2
    assert rows[0]["capital"] == "Seoul"
    assert rows[0]["ip_city"] == "Ashburn"


def test_validate_separates_valid_and_errors():
    rows = [GOOD_ROW, {**GOOD_ROW, "population": -1}]
    valid, errors = validate(rows)
    assert len(valid) == 1
    assert len(errors) == 1
    assert errors[0]["row"] == 1
