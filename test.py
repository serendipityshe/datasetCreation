from pyhocon import ConfigFactory

conf = ConfigFactory.parse_file('conf/config.conf')

app_name = conf.get_string('application.name')
debug_status = conf.get_bool('application.debug')
sever_port = conf.get_int('sever.port')

print(f'App Name: {app_name}')
print(f'Debug Status: {debug_status}')
print(f'Server Port: {sever_port}')


conf1 = ConfigFactory.parse_file('utils/dataEnhancement/conf/format.conf')
yolo_txt = conf1.get('format_name.yolo_txt')
print(yolo_txt[0])
print(len(yolo_txt))

