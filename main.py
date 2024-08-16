import pandas as pd
from constants import GRAVITY, AIR_DENSITY, C_DRAG

class RocketModel:
    def __init__(self, burn_time, m_dry, m_wet, thrust=4500, area=0.1):
        self.burn_time = burn_time
        self.m_dry = m_dry
        self.m_wet = m_wet
        self.thrust = thrust
        self.area = area
        self.initial_velocity = 0  # m/s
        self.initial_altitude = 0  # m
        self.time_increment = 0.10  # seconds
        self.results = []

    def f_drag(self, v):
        return 0.5 * C_DRAG * AIR_DENSITY * self.area * v**2

    def weight(self, m):
        return m * GRAVITY

    def acceleration(self, f_net, m):
        return f_net / m

    def velocity(self, v0, a, t):
        return v0 + a * t

    def altitude(self, h0, v0, t, a):
        return h0 + v0 * t + 0.5 * a * t**2

    def simulate(self):
        current_time = 0
        current_mass = self.m_wet 
        current_mass = current_mass -(prop_mass / (current_mass / burn_time)) if current_time < burn_time else m_wet - prop_mass

        while current_time <= self.burn_time:
            # Calculate thrust and drag force at current step

            f_drag_current = self.f_drag(self.initial_velocity)
            
            # Calculate net force and acceleration
            f_thrust = self.thrust

            f_net = f_thrust - f_drag_current - self.weight(current_mass)
            a = self.acceleration(f_net, current_mass)
            # f_thrust = a * current_time +(f_thrust)
            
            # Calculate velocity and altitude
            v = self.velocity(self.initial_velocity, a, self.time_increment)
            h = self.altitude(self.initial_altitude, self.initial_velocity, self.time_increment, a)

            # Store the results for the current time step
            self.results.append({
                "Time": current_time,
                "Thrust": f_thrust,
                "Drag Force": f_drag_current,
                "Weight": self.weight(current_mass),
                "Acceleration": a,
                "Velocity": v,
                "Altitude": h,
            })

            # Update initial conditions for the next step
            self.initial_velocity = v
            self.initial_altitude = h
            current_time += self.time_increment

        return self.results

    def print_results(self):
        for result in self.results:
            print(f"Time: {result['Time']:.2f} s, Thrust: {result['Thrust']} N, Drag Force: {result['Drag Force']:.2f} N, "
                  f"Weight: {result['Weight']:.2f} N, Acceleration: {result['Acceleration']:.2f} m/s², "
                  f"Velocity: {result['Velocity']:.2f} m/s, Altitude: {result['Altitude']:.2f} m")

if __name__ == "__main__":
    # Sample inputs
    df = pd.read_csv('data/Hypertek_L550.csv')
    m_dry = 25  # in kg
    m_wet = 48  # in kg
    prop_mass = 23

    burn_time = 10  # Extract the last value as the total burn time


    # Create RocketModel instance
    rocket = RocketModel(burn_time=burn_time, m_dry=m_dry, m_wet=m_wet)

    # Run simulation
    rocket.simulate()

    # Print results
    rocket.print_results()
