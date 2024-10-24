import pickle 

with open("seen_imgs.pkl", "rb") as f: 
    seen_imgs = pickle.load(f) 

print(f"{seen_imgs = }")
print(f"{len(seen_imgs) = }")