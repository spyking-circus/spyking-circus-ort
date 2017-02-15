##### Local version
import circusort


director = circusort.create_director()

manager = director.create_manager()
# or
# manager = director.create_manager(address='local')

reader = manager.create_reader()
computer = manager.create_computer()
writer = manager.create_writer()

reader.configure(path="/tmp/reader.txt")
# reader.input.configure()
reader.output.configure(protocol='tcp', interface='*')

computer.configure()
computer.input.configure()
# or
computer.input.connect(reader.output)
computer.output.configure()

writer.configure(path="/tmp/writer.txt")
writer.input.configure()
# or
writer.input.connect(computer.output)
# writer.output.configure()

# reader.initialize()
# computer.initialize()
# writer.initialize()
# # or
# manager.initialize_all()
# # or
director.initialize_all()

# manager.start_all()
# # or
director.start_all()

director.sleep(duration=10.0)

manager.stop_all()
# or
director.stop_all()

director.destroy_all()
