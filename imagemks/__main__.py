if __name__ == '__main__':

    import sys

    commands = [
        'download',
        'info'
    ]

    if len(sys.argv) == 1:
        print("Available commands: " + ', '.join(commands))

    else:
        command = sys.argv[1]

        if command in commands:
            print('ImageMKS %s'%command)
        else:
            print("Available commands: " + ', '.join(commands))
