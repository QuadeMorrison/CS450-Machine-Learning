class myClass(object):

    def __init__(self, x):
        # python doesnt have private variables the underline tells
        # people you dont want them to touch the variable
        self._doesntMatterWhatImCalled = x

    @property
    def x(self):
        return self._doesntMatterWhatImCalled

    @x.setter
    def x(self, x):
        self._doesntMatterWhatImCalled = x + 1


def main():
    myObject = myClass(3)
    print(myObject.x)
    myObject.x = 1
    print(myObject.x)

main()
