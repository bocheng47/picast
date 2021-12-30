from pigpio_dht import DHT22
import time

dht_gpio = 4 # BCM Numbering

sensor = DHT22(dht_gpio)

while True:
    try:
        result = sensor.read()
        print(result)
        
        if result["valid"] == True:
            
            temperature_c = result["temp_c"]
            humidity = result["humidity"]
            
            print(
                "Temp: {:.1f} C ,   Humidity: {}% ".format(
                    temperature_c, humidity
                )
            )
        
    except RuntimeError as error:
        # Errors happen fairly often, DHT's are hard to read, just keep going
        print(error.args[0])
        time.sleep(2.0)
        continue
    except Exception as error:
        dhtDevice.exit()
        raise error

    time.sleep(1.0)