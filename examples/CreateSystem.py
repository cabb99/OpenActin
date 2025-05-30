
import coarseactin

# Create the system
s = coarseactin.system()

actin = openactin.Actin()
camkii = openactin.CaMKII(some properties)

# Add the system components
s.add_molecule(openactin.actin,20,...)
s.add_molecule(openactin.camkii,20,...)

# List elements
print(s.molecules)

# Remove elements
s.remove(range(25,30))

# Add forces
s.add_force()
s.add_force(camkii.forces) #ntains actin to camkii forces

#List forces
print(s.forces)

#Remove forces

# Print forces
print(s.forces)

# Add forces
s.add_forces()
s.add_nematic_constraint()

# Simulate the system
s.initialize_MD(output='test')
s.run(10)




