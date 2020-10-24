import unittest
from genetic_algorithm import genetic_algorithm

class Square:
    def __init__(self, side):
        """ creates a square having the given side
        """
        self.side = side

    def area(self):
        """ returns area of the square
        """
        if self.side < 0 : return -1
        return self.side ** 2

    def perimeter(self):
        """ returns perimeter of the square
        """
        return 4 * self.side

    def __repr__(self):
        """ declares how a Square object should be printed
        """
        s = 'Square with side = ' + str(self.side) + '\n' + \
            'Area = ' + str(self.area()) + '\n' + \
            'Perimeter = ' + str(self.perimeter())
        return s


class TestContainer(unittest.TestCase):
    longMessage = True

def make_test_function(description, a, b) :

    def test(self) :
        sq = Square(a)
        self.assertEqual(sq.area(), b,
                        f"Area is shown {sq.area()} rather than {b}")

    return test

if __name__ == '__main__':

    #Automated Test Generation
    
    for i in range(3) :
        side, area = genetic_algorithm(20)
        test_func = make_test_function(i+1, side, area)
        setattr(TestContainer, "test_{}".format(i+1), test_func)

    unittest.main()