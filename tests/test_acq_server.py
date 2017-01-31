import circusort



config = circusort.io.load_configuration()
interface = config.acquisition.interface
port = config.acquisition.port

circusort.acq.spawn_server(interface, port)
