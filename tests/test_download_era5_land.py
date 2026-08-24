import tempfile
import unittest
from pathlib import Path

from netCDF4 import Dataset

from scripts.download_era5_land import (
    CDS_FIELD_LIMIT,
    DownloadTask,
    build_tasks,
    download_task,
    expected_grid_shape,
    validate_netcdf,
)


class FakeClient:
    def __init__(self):
        self.calls = []

    def retrieve(self, dataset, request, target):
        self.calls.append((dataset, request, target))
        task = DownloadTask(
            variable="t2m",
            year=int(request["year"][0]),
            month=int(request["month"][0]),
            area=tuple(request.get("area", ())),
        )
        latitude_count, longitude_count = expected_grid_shape(task.area)
        with Dataset(target, "w") as output:
            output.createDimension("time", task.field_count)
            output.createDimension("latitude", latitude_count)
            output.createDimension("longitude", longitude_count)
            output.createVariable("time", "i4", ("time",))
            output.createVariable("latitude", "f4", ("latitude",))
            output.createVariable("longitude", "f4", ("longitude",))
            output.createVariable("t2m", "f4", ("time", "latitude", "longitude"))


class TestDownloadEra5Land(unittest.TestCase):
    def test_baseline_plan_is_below_field_limit(self):
        tasks = build_tasks(1991, 2018, ["t2m", "d2m", "sp", "ssrd", "strd", "tp"])

        self.assertEqual(len(tasks), 28 * 12 * 6)
        self.assertLessEqual(max(task.field_count for task in tasks), CDS_FIELD_LIMIT)
        self.assertEqual(tasks[0].filename, "t2m.199101.nc")
        self.assertEqual(tasks[-1].filename, "tp.201812.nc")

    def test_month_subset(self):
        tasks = build_tasks(1991, 1991, ["t2m"], months=[2])

        self.assertEqual([task.filename for task in tasks], ["t2m.199102.nc"])

    def test_request_uses_only_valid_leap_month_days(self):
        task = DownloadTask("tp", 1992, 2)

        request = task.request()

        self.assertEqual(request["day"][0], "01")
        self.assertEqual(request["day"][-1], "29")
        self.assertEqual(task.field_count, 29 * 24)
        self.assertEqual(request["variable"], ["total_precipitation"])
        self.assertEqual(request["data_format"], "netcdf")

    def test_area_shape_is_inclusive(self):
        self.assertEqual(expected_grid_shape((1.0, 2.0, 0.8, 2.3)), (3, 4))

    def test_download_validates_and_resumes_at_file_level(self):
        task = DownloadTask("t2m", 1991, 1, (0.1, 0.0, 0.0, 0.1))
        client = FakeClient()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / task.filename

            first = download_task(client, task, destination, attempts=1, retry_delay=0)
            second = download_task(client, task, destination, attempts=1, retry_delay=0)

            self.assertEqual(first, "downloaded")
            self.assertEqual(second, "skipped")
            self.assertEqual(len(client.calls), 1)
            validate_netcdf(destination, task)

    def test_invalid_existing_file_is_replaced(self):
        task = DownloadTask("t2m", 1991, 1, (0.1, 0.0, 0.0, 0.1))
        client = FakeClient()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / task.filename
            destination.write_bytes(b"invalid")

            status = download_task(client, task, destination, attempts=1, retry_delay=0)

            self.assertEqual(status, "downloaded")
            self.assertTrue(
                destination.with_name(destination.name + ".invalid").exists()
            )
            validate_netcdf(destination, task)


if __name__ == "__main__":
    unittest.main()
