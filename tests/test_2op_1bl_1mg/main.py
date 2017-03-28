# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import numpy




def repeat_test(nb_trials=5, **kwds):
    '''
    TODO add docstring

    Parameters
    ----------
    nb_trials: integer
    kwds: dictionary

    Return
    ------
    t_comp: array

    '''

    t_comp = [single_test(**kwds) for trial in range(0, nb_trials)]
    t_comp = numpy.array(t_comp)

    return t_comp


def single_test(host='127.0.0.1', size=2, nb_samples=100, data_type='float32',
    nb_buffer=1000):
    '''
    TODO complete docstring

    Parameters
    ----------
    host: string
    size: integer
        Square root of the number of electrodes for the buffer.
    nb_samples: integer
        Number of time samples for the buffer.
    data_type: string
        Data type of the buffer.
    nb_buffer: integer
        Number of buffers to process.

    Return
    ------
    t_comp: float
        Computational time.

    '''

    # 1. Create director
    interface = circusort.utils.find_interface_address_towards(host)
    director = circusort.create_director(interface=interface)
    # 2. Create manager
    manager = director.create_manager(host=host)
    # 3. Create block with read & send operations
    reader = manager.create_block('reader')
    # 4. Create block with two operations (serial composition)
    computer = manager.create_block('computer_1_2')
    # 5. Create block with receive & write operations
    writer = manager.create_block('writer')
    # 6. Configure blocks
    reader.size = size
    reader.nb_samples = nb_samples
    reader.dtype = data_type
    reader.force = True
    # 7. Initialize blocks
    reader.initialize()
    computer.initialize()
    writer.initialize()
    # 8. Connect blocks
    manager.connect(reader.output, computer.input)
    computer.configure()
    manager.connect(computer.output, writer.input)
    # 9. Connect block again
    writer.connect()
    computer.connect()
    reader.connect()
    # 10. Start blocks
    writer.start()
    computer.start()
    reader.start()
    # 11. Wait blocks stop
    reader.join()
    computer.join()
    writer.join()
    # 12. Retrieve computational time
    t_comp = writer.t_comp

    return t_comp
