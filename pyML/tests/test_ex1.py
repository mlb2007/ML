import ex1
import unittest


class TestEx1(unittest.TestCase):
    def test_a(self):
        f = ex1.Foo()
        assert f.bar() == 1
