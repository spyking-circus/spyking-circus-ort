import circusort



config = circusort.io.load_configuration()
interface = config.acquisition.server_interface
circusort.acq.spawn_server(interface)
