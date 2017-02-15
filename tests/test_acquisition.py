from multiprocessing import Process, Queue
import time
import zmq



class Manager(object):
    def __init__(self):
        self.processes = dict()

    def add_process(self, target=None, name=None, args=(), kwargs={}):
        process = Process(target=target, name=name, args=args, kwargs=kwargs)
        self.processes[name] = process
        return

    # TODO uncomment...
    # def start_processes(self):
    #     # Start processes (in random order)...
    #     for process in self.processes.itervalues():
    #         process.start()
    #     return
    # TODO remove...
    def start_processes(self):
        # Start input process...
        self.processes['input'].start()
        time.sleep(1.0)
        # Start output process...
        self.processes['output'].start()
        return


def input_process(port=None):
    # Arguments (bind)
    protocol = "tcp"
    interface = "*"
    address = "{}://{}:{}".format(protocol, interface, port)
    # Bind
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(address)
    # Wait
    time.sleep(1.1)
    for reqnum in range(0, 5):
        # Arguments (send)
        topic = 0
        data = "Bye!"
        message = "{} {}".format(topic, data)
        # Send
        socket.send(message)
        print("[ input] send data {}: {}".format(reqnum, data))
        # Wait
        time.sleep(0.000001)
    return

def output_process(port=None):
    # Arguments (connect)
    protocol = "tcp"
    interface = "localhost"
    address = "{}://{}:{}".format(protocol, interface, port)
    # Connect
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(address)
    # Subscribe
    topic = 0
    socket.setsockopt(zmq.SUBSCRIBE, str(topic))
    for reqnum in range(0, 5):
        # Receive
        print("[output] wait data")
        message = socket.recv()
        topic, data = message.split()
        print("[output] recv data {}: {}".format(reqnum, data))
    return


def read(path, connection):
    f = open(path, 'r')
    for k in nb_chunks:
        chunk = f.read(1024)
        connection.send(chunk_index)
    f.close()
    return

def main():
    inbox = Connection()
    outbox = Connection()
    reader = Process(target=read, kwargs=reader_kwargs)

def main():
    in_max_size = 10
    out_max_size = 10
    # Define connection
    inbox = Queue(in_max_size)
    outbox = Queue(out_max_size)
    # Process to read
    reader_kwargs = {
        'path': path,
        'inbox': inbox
    }
    reader = Process(target=read, kwargs=reader_kwargs)
    reader.start()
    # Process to compute
    computer_kwargs = {
        'inbox': inbox,
        'outbox': outbox
    }
    computer = Process(target=compute, kwargs=computer_kwargs)
    computer.start()
    # Process to write
    writer_kwargs = {
        'outbox': outbox,
        'path': path
    }
    writer = Process(target=write, kwargs=writer_kwargs)
    writer.start()



if __name__ == '__main__':

    port = 5557

    manager = Manager()

    # process_1 = manager.add_process(target=input_process, name='input', kwargs={'port': port})
    # process_2 = manager.add_process(target=output_process, name='output', kwargs={'port': port})
    manager.add_process(target=input_process, name='input', kwargs={'port': port})
    manager.add_process(target=output_process, name='output', kwargs={'port': port})

    manager.start_processes()

    # manager.bind_processes()
    #
    # manager.connect_processes()
    #
    # manager.run_processes()
