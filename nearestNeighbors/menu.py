import os

class Menu(object):

    def __init__(self, options):
        self.options = options
        self.options["q"] = { "name" : "Quit",
                              "value" : None,
                              "handler" : self.quit }

    def input(self):
        print(self)
        option = input()
        os.system("cls")
        os.system("clear")
        if self.options.get(option):
            value = self.options[option]["handler"](self, self.options[option]["value"])
            self.options[option]["value"] = value

    def start(self):
        while not self.options["q"]["value"]:
            self.input()

    def quit(self, menu, value):
        os.system("cls")
        os.system("clear")
        return True

    def getValue(self, input):
        return self.options[input]["value"]

    def __str__(self):
        help = "Type one of the following options (denoted by square brackets) and hit enter\n\n"
        for key in self.options:
            #help += "\t[input] " + option["name"]
            help += "[" + key + "] " + self.options[key]["name"]
            if self.options[key]["value"]:
                help += " : " + self.options[key]["value"]
            help += "\n"

        return help
