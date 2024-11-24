#!/usr/bin/env python3
import os
import sys

if __name__ == '__main__':
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if pkg_dir not in sys.path:
        sys.path.append(pkg_dir)
    from serveur.main import main
    main()
