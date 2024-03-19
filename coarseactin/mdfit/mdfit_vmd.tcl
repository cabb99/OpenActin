# Helper function to peek at the next line of the file without consuming it
proc peekNextLine {fileId} {
    # Save the current position
    set current_pos [tell $fileId]
    # Read the next line
    set line [gets $fileId]
    # Return to the saved position
    seek $fileId $current_pos
    # Return the line read
    return $line
}

proc read_forces {filename} {
    # Open the file for reading
    set file [open $filename r]

    # Initialize an empty list to hold forces; this will be a list of lists
    set forces {}

    # Read the file line by line
    while {[gets $file line] >= 0} {
        # Check if the line starts with "Frame"; if so, start a new frame
        if {[string match "Frame*" $line]} {
            # Start a new frame: initialize an empty list for this frame's forces
            set current_frame_forces {}
        } else {
            # Split the line into components: atom index and force components
            set line_data [split $line " "]
            set force_x [lindex $line_data 1]
            set force_y [lindex $line_data 2]
            set force_z [lindex $line_data 3]
            
            # Append the force components to the current frame's list
            lappend current_frame_forces [list $force_x $force_y $force_z]
        }
        # At the end of a frame, append the frame's forces to the main list
        if {[eof $file] || [string match "Frame*" [peekNextLine $file]]} {
            echo [llength $current_frame_forces]
            echo [llength $forces]
            lappend forces $current_frame_forces
            echo [llength $current_frame_forces]
            echo [llength $forces]
        }
    }
    # Append the last frame's forces after exiting the loop
    if {[llength $current_frame_forces] > 0} {
        echo [llength $current_frame_forces]
        echo [llength $forces]
        lappend forces $current_frame_forces
        echo [llength $current_frame_forces]
        echo [llength $forces]
    }
    # Close the file
    close $file
    # Return the list of forces
    return $forces
}

# Procedure to plot forces for a specified frame
proc plot_forces_for_frame {forces frame_idx scale_factor arrow_color} {
    set mol top
    
    # Clear previous graphics
    graphics $mol delete all
    
    # Get the list of forces for the specified frame
    set frame_forces [lindex $forces $frame_idx]

    # Iterate over each atom's force vector in the frame
    foreach force $frame_forces {
        # force is a list containing the x, y, z components of the force vector
        set force_x [lindex $force 0]
        set force_y [lindex $force 1]
        set force_z [lindex $force 2]
        
        # Assume the atom index corresponds to the iteration index (adjust if needed)
        set atom_idx [expr {[lsearch $frame_forces $force]}]
        
        # Get the position of the atom
        set pos [lindex [[atomselect $mol "index $atom_idx"] get {x y z}] 0]
        
        # Calculate the end position of the arrow based on the force vector
        set end_pos [list [expr {[lindex $pos 0] + $scale_factor * $force_x}] \
                           [expr {[lindex $pos 1] + $scale_factor * $force_y}] \
                           [expr {[lindex $pos 2] + $scale_factor * $force_z}]]
        
        # Draw the arrow (cylinder + cone for the arrowhead)
        graphics $mol color $arrow_color
        graphics $mol cylinder $pos $end_pos radius 0.1
        graphics $mol cone $end_pos [list [expr {[lindex $end_pos 0] + 0.1 * $force_x}] \
                                       [expr {[lindex $end_pos 1] + 0.1 * $force_y}] \
                                       [expr {[lindex $end_pos 2] + 0.1 * $force_z}]] radius 0.2

    }
    
    # Update the VMD scene to display the new graphics
    display update
    update
}

set forces [read_forces "forces.txt"]
plot_forces_for_frame $forces 0 0.1 "red" ;# Plot forces for frame 0 with scale factor 0.1 and red arrows