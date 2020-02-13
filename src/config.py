import configparser

config = configparser.ConfigParser()
config.read('project.conf')

#print(config.getint('parameter', 'max_episodes'))
