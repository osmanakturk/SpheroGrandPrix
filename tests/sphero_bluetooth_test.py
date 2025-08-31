from spherov2 import scanner
from spherov2.toy.bolt import BOLT
from spherov2.adapter.bleak_adapter import BleakAdapter
import time



toys = scanner.find_toys(timeout=10)
bolts = []


print(f"Total: {len(toys)}")

for toy in toys:
    try:
        print("*"*20)
        print(f"Toy: {toy}")
        print(f"Toy Name: {toy.name}")
        print(f"Toy Address: {toy.address}")
        print(f"Toy Type: {toy.toy_type}")
        print("*"*20)
        bolt = BOLT(toy, BleakAdapter)
        bolt.__enter__()
        time.sleep(0.1)
        bolts.append(bolt)
        print(f"Bolt Added: {toy.name}")
    except Exception as e:
        print(f"Bolt Not Added: {toy.name}, e: {e}")


try:

    for bolt in bolts:
        try:
            bolt.wake()
        except Exception as e:
            print(e)
        time.sleep(0.2)
        bolt.reset_yaw()
        bolt.set_stabilization(True)


    for bolt in bolts:
        bolt.drive_with_heading(speed=255, heading=0, drive_flags=0)

    time.sleep(2)

    for bolt in bolts:
        bolt.drive_with_heading(speed=0, heading=0, drive_flags=0)

except Exception as e:
    print(e)
finally:
    for bolt in bolts:
        bolt.__exit__(None, None, None)

