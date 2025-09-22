import random

def generate_timetables(n_examples=50, n_trains=8, n_platforms=4, max_time=40):
    all_timetables = []
    for _ in range(n_examples):
        trains = []
        for i in range(n_trains):
            arrival = random.randint(0, max_time-10)
            departure = random.randint(arrival+5, min(arrival+15, max_time))
            priority = random.randint(1, 5)
            trains.append({"id": f"T{i+1}", "arrival": arrival, "departure": departure, "priority": priority})
        platforms = [f"Platform {i+1}" for i in range(n_platforms)]
        all_timetables.append({"trains": trains, "platforms": platforms})
    return all_timetables

if __name__ == "__main__":
    timetables = generate_timetables()
    # Save for later evaluation
    import pickle
    with open("random_timetables.pkl", "wb") as f:
        pickle.dump(timetables, f)
    print("Generated 50 random timetables")