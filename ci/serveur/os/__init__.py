class OS:
    def __init__(self, name):
        self.name = name


debian_sid = OS('debian')
arch_linux = OS('archlinux')
all_os = [debian_sid, arch_linux]
