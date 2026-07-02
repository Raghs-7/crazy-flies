import time
import cflib.crtp

from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.log import LogConfig
from cflib.positioning.position_hl_commander import PositionHlCommander


uris = {
    'radio://0/80/2M/E7E7E7E701',
    'radio://0/80/2M/E7E7E7E702'
}


positions = {}


def start_position_logging(scf):

    uri = scf.cf.link_uri

    log_conf = LogConfig(
        name=f'Position_{uri[-3:]}',
        period_in_ms=500
    )

    log_conf.add_variable('stateEstimate.x', 'float')
    log_conf.add_variable('stateEstimate.y', 'float')
    log_conf.add_variable('stateEstimate.z', 'float')

    def log_callback(timestamp, data, logconf):

        positions[uri] = (
            round(data['stateEstimate.x'], 2),
            round(data['stateEstimate.y'], 2),
            round(data['stateEstimate.z'], 2)
        )

    scf.cf.log.add_config(log_conf)

    log_conf.data_received_cb.add_callback(log_callback)

    log_conf.start()


def takeoff_hover_land(scf):

    uri = scf.cf.link_uri

    try:

        with PositionHlCommander(
            scf,
            default_height=1.0,
            controller=PositionHlCommander.CONTROLLER_PID
        ) as pc:

            print(f"{uri} taking off")

            # hover safely
            time.sleep(3)

            print(f"{uri} landing")

        print(f"{uri} landed safely")

    except Exception as e:

        print(f"{uri} ERROR: {e}")

        try:
            scf.cf.commander.send_stop_setpoint()
        except:
            pass


if __name__ == '__main__':

    cflib.crtp.init_drivers()

    factory = CachedCfFactory(rw_cache='./cache')

    with Swarm(uris, factory=factory) as swarm:

        print("Connected to Crazyflies")

        # allow TWR stabilization
        time.sleep(15)

        print("Starting position logging")

        swarm.parallel_safe(start_position_logging)

        # wait for coordinates
        time.sleep(5)

        print("\nSPAWN LOCATIONS:\n")

        for uri, pos in positions.items():
            print(f"{uri} -> {pos}")

        print("\nStarting flight\n")

        swarm.parallel_safe(takeoff_hover_land)

        print("\nMission complete")