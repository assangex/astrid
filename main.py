import pandas as pd
import numpy as np
import logging
import os
import math
import matplotlib.pyplot as plt
from data.constants import GRAVITY
from libs.logo import astrid_logo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def clamp(value, min_val, max_val):
    return max(min(value, max_val), min_val)

class Astrid:
    def __init__(self, df: pd.DataFrame, thrust_data: pd.Series, burn_time: float, m_wet: float, of_ratio: float, prop_frac: float, area: float, time_inc: float = 0.10, launch_altitude: float = 0, cd_data: pd.DataFrame = None):
        self.burn_time = burn_time
        self.of_ratio = of_ratio
        self.prop_frac = prop_frac
        self.density = df["Density(kg/m^3)"].tolist()
        self.m_wet = m_wet
        self.thrust_data = thrust_data
        self.area = area
        self.time_inc = time_inc
        self.i_altitude = launch_altitude
        self.cd_data = cd_data
        self.results = []

        self.max_mach = 0
        self.max_altitude = 0
        self.max_acceleration = 0
        self.max_g_force = 0
        self.max_q = 0
        self.max_velocity = 0
        self.apogee_time = 0

        self.total_prop_mass = self.prop_frac * m_wet
        self.fuel_mass = self.total_prop_mass / (1 + self.of_ratio)
        self.ox_mass = self.total_prop_mass - self.fuel_mass
        self.current_mass = m_wet

        self.mass_dot = self.total_prop_mass / burn_time
        self.isp = self.thrust_data.mean() / (self.mass_dot * GRAVITY)
        self.total_impulse = self.thrust_data.sum() * self.time_inc
        self.average_thrust = self.thrust_data.mean()

        self.main_cd = 1.9
        self.main_diameter = 3.6576
        self.drogue_cd = 0.97
        self.drogue_diameter = 0.9144
        self.main_area = math.pi * (self.main_diameter / 2) ** 2
        self.drogue_area = math.pi * (self.drogue_diameter / 2) ** 2

        logging.info(f"Specific Impulse (Isp): {self.isp:.2f} seconds")
        logging.info(f"Mass Flow Rate: {self.mass_dot:.4f} kg/s")
        logging.info(f"Total Impulse: {self.total_impulse:.2f} Ns")
        logging.info(f"Average Thrust: {self.average_thrust:.2f} N")

    def get_cd(self, time: float, altitude: float, mach: float) -> float:
        if time < self.burn_time:
            cd = np.interp(mach, self.cd_data["Mach"], self.cd_data["CD"])
        else:
            # After burn time, determine the Cd based on altitude
            if altitude > 1250:
                cd = self.drogue_cd
            else:
                cd = self.main_cd

        return cd

    def drag(self, altitude: float, velocity: float, density: float, time: float) -> float:
        mach_value = self.mach(velocity)
        cd = self.get_cd(time, altitude, mach_value)
        if velocity < 0:
            if altitude <= 1000:
                return -0.5 * self.main_cd * density * velocity**2 * self.main_area
            elif altitude <= 1375:
                return -0.5 * self.drogue_cd * density * velocity**2 * self.drogue_area
        return 0.5 * cd * density * velocity**2 * self.area

    def weight(self) -> float:
        return self.current_mass * GRAVITY

    def acceleration(self, force: float) -> float:
        return force / self.current_mass

    def velocity(self, v0: float, a: float) -> float:
        return v0 + a * self.time_inc

    def altitude(self, h0: float, v0: float, a: float) -> float:
        return h0 + v0 * self.time_inc + 0.5 * a * self.time_inc**2

    def get_density(self, altitude: float) -> float:
        index = clamp(int(altitude // 50), 0, len(self.density) - 1)
        return self.density[index]

    def ve(self) -> float:
        return self.isp * GRAVITY

    def f_thrust(self, current_time: float) -> float:
        index = clamp(int(current_time // self.time_inc), 0, len(self.thrust_data) - 1)
        return self.thrust_data.iloc[index]

    def mach(self, velocity: float, speed_of_sound: float = 343) -> float:
        return velocity / speed_of_sound

    def maxq(self, density: float, velocity: float) -> float:
        return 0.5 * density * velocity**2

    def simulate(self):
        current_time = 0
        current_velocity = 0
        current_altitude = self.i_altitude
        main_deployed = False
        drogue_deployed = False

        self.results.append({
            "Time": current_time,
            "Net Force": 0,
            "Drag Force": 0,
            "Thrust": 0,
            "Weight": self.weight(),
            "Acceleration": 0,
            "Velocity": current_velocity,
            "Altitude": current_altitude,
            "Density": self.get_density(current_altitude),
            "Mach": 0,
            "Max Q": 0,
            "Status": "Climbing"
        })

        while True:
            if current_time < self.burn_time:
                fuel_consumed = (self.fuel_mass / self.burn_time) * self.time_inc
                ox_consumed = (self.ox_mass / self.burn_time) * self.time_inc
                self.current_mass -= (fuel_consumed + ox_consumed)
            else:
                self.current_mass = self.m_wet - self.total_prop_mass

            density = self.get_density(current_altitude)
            f_drag = self.drag(current_altitude, current_velocity, density, current_time)
            f_net = self.f_thrust(current_time) - f_drag - self.weight()
            a = self.acceleration(f_net)
            current_velocity = self.velocity(current_velocity, a)
            mach_value = self.mach(current_velocity)
            current_altitude = self.altitude(current_altitude, current_velocity, a)
            maxq = self.maxq(density, current_velocity)
            status = "Climbing" if current_altitude > self.i_altitude else "Descending"

            if not main_deployed and current_altitude <= 1000 and current_velocity < 0:
                status = "Main Parachute Deployed"
                main_deployed = True

            if not drogue_deployed and current_altitude <= 1375 and current_velocity < 0:
                status = "Drogue Parachute Deployed"
                drogue_deployed = True

            if current_altitude > self.max_altitude:
                self.max_altitude = current_altitude
                self.apogee_time = current_time

            self.max_mach = max(self.max_mach, mach_value)
            self.max_acceleration = max(self.max_acceleration, a)
            self.max_velocity = max(self.max_velocity, current_velocity)
            g_force = a / GRAVITY
            self.max_g_force = max(self.max_g_force, g_force)
            self.max_q = max(self.max_q, maxq)

            self.results.append({
                "Time": current_time,
                "Net Force": f_net,
                "Drag Force": f_drag,
                "Thrust": self.f_thrust(current_time),
                "Weight": self.weight(),
                "Acceleration": a,
                "Velocity": current_velocity,
                "Altitude": current_altitude,
                "Density": density,
                "Mach": mach_value,
                "Status": status
            })

            current_time += self.time_inc

            if status == "Descending" and current_altitude < 0:
                break

        self.m_dry = self.current_mass
        logging.info(f"Final Dry Mass: {self.m_dry:.2f} kg")
        logging.info(f"Time of Apogee: {self.apogee_time:.2f} seconds")

    def save(self, filename: str):
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(filename, index=False)
        logging.info(f"Results saved to {filename}")
        if os.path.exists(filename):
            logging.info(f"File size: {os.path.getsize(filename) / 1024:.2f} KB")

    def plot(self):
        times = [result['Time'] for result in self.results]
        altitudes = [result['Altitude'] for result in self.results]
        thrusts = [result["Thrust"] for result in self.results]
        velocities = [result['Velocity'] for result in self.results]
        accelerations = [result['Acceleration'] for result in self.results]

        plt.figure(figsize=(10, 8))

        plt.subplot(3, 1, 1)
        plt.plot(times, altitudes, label='Altitude (m)', color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Altitude (m)')
        plt.title('Rocket Altitude over Time')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(times, velocities, label='Velocity (m/s)', color='green')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Rocket Velocity over Time')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(times, thrusts, label='Thrust (N)', color='purple')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²) / Thrust (N)')
        plt.title('Thrust over Time')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def print_results(self):
        for result in self.results:
            logging.info(f"Time: {result['Time']:.2f} s, Force: {result['Net Force']:.2f} N, Density: {result['Density']:.3f} kg/m³, "
                         f"Drag Force: {result['Drag Force']:.2f} N, Weight: {result['Weight']:.2f} N, "
                         f"Acceleration: {result['Acceleration']:.2f} m/s², Velocity: {result['Velocity']:.2f} m/s (Mach {result['Mach']:.2f}), "
                         f"Altitude: {result['Altitude']:.2f} m, Status: {result['Status']}")

        astrid_logo()

        logging.info(f"\nMax Mach: {self.max_mach:.2f}")
        logging.info(f"Apogee: {self.max_altitude:.2f} m at {self.apogee_time:.2f} seconds")
        logging.info(f"Max Acceleration: {self.max_acceleration:.2f} m/s²")
        logging.info(f"Max G's: {self.max_g_force:.2f} g")
        logging.info(f"Max Q: {self.max_q:.2f} Pa")
        logging.info(f"Max Velocity: {self.max_velocity:.2f} m/s")
        logging.info(f"Final Dry Mass: {self.m_dry:.2f} kg")
        logging.info(f"Mass Flow Rate: {self.mass_dot:.4f} kg/s")
        logging.info(f"Ox Mass: {self.ox_mass:.2f} kg")
        logging.info(f"Fuel Mass: {self.fuel_mass:.2f} kg")
        logging.info(f"Exhaust Velocity: {self.ve():.2f} m/s")
        logging.info(f"Impulse: {self.total_impulse:2f}")

def adjust_data_length(data: pd.Series, target_length: int) -> pd.Series:
    actual_length = len(data)
    
    if actual_length < target_length:
        # If the data points are fewer, interpolate to match the points
        data = np.interp(
            np.linspace(0, actual_length - 1, target_length),
            np.arange(actual_length),
            data
        )
    elif actual_length > target_length:
        # If the data points are more, truncate to match the points
        data = data.iloc[:target_length]

    return pd.Series(data)

if __name__ == "__main__":
    try:
        df = pd.read_csv('/home/sabir/Apps/Code/code/projects/astrid/data/data.csv').fillna(0)
        cd_data = pd.read_csv('/home/sabir/Apps/Code/code/projects/astrid/data/variable_CD.csv')
        thrust_df = pd.read_csv('/home/sabir/Apps/Code/code/projects/astrid/data/thrustcurve.csv')
        thrust_df['New Thrust'] = thrust_df['Thrust'] * 4

        burn_time = 12
        thrust = 5500
        m_wet = 65
        of_ratio = 5.5
        prop_frac = 0.53
        area = math.pi * (8 * 0.0254 / 2) ** 2
        launch_altitude = 1350
        time_inc = 0.05

        # Adjust the thrust data length to match the required number of points
        required_points = int(burn_time / time_inc)
        adjusted_thrust_data = adjust_data_length(thrust_df['New Thrust'], required_points)

        rocket = Astrid(df, burn_time=burn_time, m_wet=m_wet, thrust_data=adjusted_thrust_data, of_ratio=of_ratio, prop_frac=prop_frac, launch_altitude=launch_altitude, cd_data=cd_data, area=area, time_inc=time_inc)
        rocket.simulate()
        rocket.save('astrid.csv')
        rocket.print_results()
        rocket.plot()

    except Exception as e:
        logging.error(f"An error occurred: {e}")
