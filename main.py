import pandas as pd
from data.constants import GRAVITY, C_DRAG
import math
# 
class RocketModel:
    def __init__(self, burn_time, m_dry, m_wet, thrust=4500, area=.1016, time_set=0):
        self.burn_time = burn_time
        self.density = df['Density(kg/m^3)']
        self.time_set = time_set
        self.m_dry = m_dry
        self.m_wet = m_wet
        self.thrust = thrust
        self.area = area  # m^2
        self.initial_velocity = 0  # m/s
        self.initial_altitude = 0  # m
        self.time_increment = 0.10  # seconds
        self.results = []

    def f_drag(self, v, area, density):
        return C_DRAG * density * (v ** 2) * 0.5 * (area ** 2) * math.pi
    
    def weight(self, m):
        return m * GRAVITY

    def acceleration(self, f_net, m):
        return f_net / m

    def velocity(self, v0, a, t):
        return (a * t) + v0

    def altitude(self, h0, v0, t, a):
        return h0 + v0 * t + 0.5 * a * t**2
    
    def get_density(self, current_altitude):
        rounded_altitude = int(current_altitude // 100)  # Round down to nearest 100 meters
        if rounded_altitude in self.density.index:
            density = self.density.loc[rounded_altitude]
        else:
            density = self.density.iloc[-1]  # Refer to previous density if altitude exceeds known range
        return density

    def simulate(self):
        current_time = 0
        current_mass = self.m_wet
        current_altitude = self.initial_altitude  # Start from initial altitude

        while current_time <= self.time_set:
            # Update mass if burn time is still ongoing
            if current_time < self.burn_time:
                current_mass -= prop_mass / (self.burn_time / self.time_increment)

            # Grab the current density based on altitude
            density = self.get_density(current_altitude)

            # Calculate thrust and drag force at current step
            f_drag_current = self.f_drag(self.initial_velocity, self.area, density)

            # Calculate net force and acceleration
            f_thrust = self.thrust if current_time < self.burn_time else 0  # Thrust stops after burn time
            f_net = f_thrust - (f_drag_current + self.weight(current_mass))
            a = self.acceleration(f_net, current_mass)

            # Calculate velocity and altitude
            v = self.velocity(self.initial_velocity, a, self.time_increment)
            h = self.altitude(self.initial_altitude, self.initial_velocity, self.time_increment, a)

            # Store the results for the current time step
            self.results.append({
                "Time": current_time,
                "Net Force": f_net,
                "Drag Force": f_drag_current,
                "Weight": self.weight(current_mass),
                "Acceleration": a,
                "Velocity": v,
                "Altitude": h,
                "Density": density
            })

            # Update initial conditions for the next step
            self.initial_velocity = v
            self.initial_altitude = h
            current_altitude = h  # Update current altitude
            current_time += self.time_increment

        return self.results

    def print_results(self):
        for result in self.results:
            print(f"Time: {result['Time']:.2f} s, Net Force: {result['Net Force']:.2f} N, Density: {result['Density']:.3f} kg/m³, "
                  f"Drag Force: {result['Drag Force']:.2f} N, Weight: {result['Weight']:.2f} N, "
                  f"Acceleration: {result['Acceleration']:.2f} m/s², Velocity: {result['Velocity']:.2f} m/s, "
                  f"Altitude: {result['Altitude']:.2f} m")

if __name__ == "__main__":
    # Sample inputs
    df = pd.read_csv('data/data.csv')
    m_dry = 25  # in kg
    m_wet = 48  # in kg
    prop_mass = m_wet - m_dry
    burn_time = 20  # Burn time in seconds
    time_set = time_set = 30  # Total simulation time in seconds
    # Create RocketModel instance
    rocket = RocketModel(burn_time=burn_time, m_dry=m_dry, m_wet=m_wet, time_set=time_set)

    # Run simulation
    rocket.simulate()

    # Print results
    rocket.print_results()

