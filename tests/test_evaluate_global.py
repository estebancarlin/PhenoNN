import unittest

from phenonn.data.dataset_netcdf import SPLIT_CODES
from phenonn.training.train_global import parse_years


class TestEvaluateGlobal(unittest.TestCase):
    def test_explicit_split_codes_are_stable(self):
        self.assertEqual(
            SPLIT_CODES,
            {"train": 0, "validation": 1, "test": 2, "buffer": 3},
        )

    def test_year_parser(self):
        self.assertEqual(parse_years("2015-2016"), [2015, 2016])
        self.assertEqual(parse_years("2015,2017"), [2015, 2017])


if __name__ == "__main__":
    unittest.main()
