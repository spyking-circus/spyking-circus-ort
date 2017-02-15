##### Network version
import circusort


director = circusort.create_director()

manager_1 = director.create_manager(address="134.157.180.212")
manager_2 = director.create_manager(address="134.157.180.213")
manager_3 = director.create_manager(address="134.157.180.214")

reader = manager_1.create_reader()
computer = manager_2.create_computer()
writer = manager_3.create_writer()

reader.configure()
# reader.input.configure()
reader.output.configure()

computer.configure()
computer.input.configure()
# or
computer.input.connect(reader.output)
computer.output.configure()

writer.configure()
writer.input.configure()
# or
computer.input.connect(computer.output)
# writer.output.configure()

reader.initialize()
computer.initialize()
writer.initialize()
# or
manager_1.initialize_all()
manager_2.initialize_all()
manager_3.initialize_all()
# or
director.initialize_all()

manager_1.start_all()
manager_2.start_all()
manager_3.start_all()
# or
director.start_all()

director.sleep(duration=1.0)

manager_1.stop_all()
manager_2.stop_all()
manager_3.stop_all()
# or
director.stop_all()

director.destroy_all()
