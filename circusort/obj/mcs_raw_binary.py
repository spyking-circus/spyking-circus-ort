class File:

    @classmethod
    def read_header(cls, path):

        max_nb_bytes = 3000

        header = []
        nb_read_bytes = 0
        is_end_reached = False

        fid = open(path, mode='rb')
        while not is_end_reached and nb_read_bytes <= max_nb_bytes:
            char = fid.read(1)  # i.e. read one byte
            nb_read_bytes += 1
            char = char.decode('Windows-1252')
            header.append(char)
            if nb_read_bytes >= 3:
                header_end = ''.join(header[-3:])
                is_end_reached = header_end == 'EOH'
        fid.close()

        header = ''.join(header)

        return header

    @classmethod
    def parse_header(cls, header):

        # Remove EOH tag.
        header = header[:-3]

        header = header.replace('\r', '')

        lines = header.split('\n')

        header = {}
        for line in lines:
            if '=' in line:
                parts = line.split(' = ')
                header[parts[0]] = parts[1]

        # TODO parse header.

        return header

    @classmethod
    def load(cls, path):

        # TODO parse the header.
        header = cls.read_header(path)
        header = cls.parse_header(header)

        print(header)

        # TODO complete.

        return

    def __init__(self):

        pass
