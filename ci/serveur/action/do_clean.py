from serveur.action import Action


class Clean(Action):

    @classmethod
    def desc(cls):
        return "Clean up project build directory"

    def add_arguments(self, parser):
        pass

    def do_action(self):
        pass
