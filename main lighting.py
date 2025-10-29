# import libraries
import pygame
import csv
import random
import argparse
import numpy as np
from collections import defaultdict
import math
import tempfile

# lighting engine
import pygame_light2d as pl2d
from pygame_light2d import LightingEngine, PointLight, Hull

# custom library
from button import Button

""" Flag Variables """
debug = False # debug flag to display debug information like player cords
f2_pressed = False # Flag for if the f2 key has been pressed
is_won = False

""" Boot Arguments """
parser = argparse.ArgumentParser()
parser.add_argument("--wall_collisions", help="Stop moving when hitting wall 0: False, 1: True", default=1, type=int)
parser.add_argument("--shadow", help="Display the shadow 0: False, 1: True", default=1, type=int)
parser.add_argument("--debug", help="Enable debug console 0: False, 1: True", default=0, type=int)
parser.add_argument("--gravity", help="Enable gravity 0: False, 1: True", default=0, type=int)
args = parser.parse_args()

if args.debug == 1: debug = True

""" Initialize and setup pygame """
pygame.init()
pygame.display.set_caption("Blind Maze")
windowDimentions = [943, 943]
screen = pygame.display.set_mode(windowDimentions)

lights_engine = LightingEngine(screen_res=windowDimentions, native_res=windowDimentions, lightmap_res=(int(windowDimentions[0]/2.5), int(windowDimentions[1]/2.5))) # initialize the lighting engine

""" Fonts """
# Custom Fonts
KightWarrior_Font_XLarge = pygame.font.Font("assets/fonts/knight-warrior-font/KnightWarrior-w16n8.otf", 120)
KightWarrior_Font_Large = pygame.font.Font("assets/fonts/knight-warrior-font/KnightWarrior-w16n8.otf", 90)
KightWarrior_Font_Medium = pygame.font.Font("assets/fonts/knight-warrior-font/KnightWarrior-w16n8.otf", 50)

MinecraftEvenings_Font_Medium = pygame.font.Font("assets/fonts/minecraft-evenings-font/MinecraftEvenings-lgvPd.ttf", 50)

""" Colours """
bgColour = (40, 40, 40)
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)

ambientAlpha = 200
fullyTransparent = (255, 255, 255, 0)
translucentBlack = (0, 0, 0, ambientAlpha)

lights_engine.set_ambient(128, 128, 128, 128) # set ambient light

""" Player Data """
playerHeight = 18
playerWidth = 17
playerX = 0
playerY = 0
speed = 120 # IMPORTANT: Speed is now in units per second (e.g., pixels/second)
playerPos_Pause = [playerX, playerY]

""" Lights """
light = PointLight(position=(playerX, playerY), power=1., radius=250)
light.set_color(50, 100, 200, 200)
lights_engine.lights.append(light)

""" Shadow Data """
shadowRadius = 40
featheredSize = 30
currentFeatheredSize = featheredSize

""" Shadow Surface """
shadowSurface = pygame.Surface((windowDimentions), pygame.SRCALPHA) # create surface with per-pixel alpha
shadowSurface.fill(black)

""" Generate a featherd mask """
def generate_feathered_light_mask(radius, feather_width, color=(0, 0, 0)):
    # --- Create a surface for the mask ---
    size = radius * 2 # define the size
    
    # Create the surface with SRCALPHA and fill it with the desired color (RGB)
    # The initial alpha of the surface is 255 (fully opaque)
    light_mask_surface = pygame.Surface((size, size), pygame.SRCALPHA)
    light_mask_surface.fill(color) # e.g., Fill with black (0, 0, 0, 255)

    # --- Setup numpy arrays for pixel manipulation ---
    x, y = np.ogrid[-radius:radius, -radius:radius]
    dist = np.sqrt(x**2 + y**2)
    
    # Define the core radius
    core_radius = radius - feather_width
    
    # Create the 2D Alpha Channel (0=transparent, 255=opaque)
    alpha_array = np.zeros((size, size), dtype=np.uint8)
    
    # Fully clear/transparent in the core (alpha = 0)
    alpha_array[dist <= core_radius] = 0
    
    # Gradient in the feather zone (linear fade from transparent (0) to opaque (255))
    gradient_zone = (dist > core_radius) & (dist <= radius)
    
    # Calculate alpha based on distance
    # 0 at core_radius, 255 at radius
    alpha_array[gradient_zone] = 255 * (dist[gradient_zone] - core_radius) / feather_width
    
    # Fully opaque outside the radius (alpha = 255)
    alpha_array[dist > radius] = 255
    
    # --- Apply the calculated 2D alpha array directly to the surface's alpha channel ---
    pygame.surfarray.pixels_alpha(light_mask_surface)[:] = alpha_array
    
    return light_mask_surface

featherdShadowMask = generate_feathered_light_mask(shadowRadius, featheredSize)
featherdShadowRect = featherdShadowMask.get_rect()

""" Maze Settings """
cellSize = 23 # the size of each cell in the window
rects = [] # This list will hold all the created rects
mazeDir = "mazeGen/mazes" # the directory where the maze files a stored
numOfMazes = 500 # the total number of mazes

""" Load sprite sheet """
# spritesheet = lights_engine.load_texture("assets/player/PNG/Slime1/Without_shadow/Slime1_Idle_without_shadow.png")

""" Load Sprite """
character_idle_sheet = pygame.image.load("assets/player/PNG/Slime1/Without_shadow/Slime1_Idle_without_shadow.png") # load sprite sheet
character_idle = character_idle_sheet.subsurface(pygame.Rect(280, 22, playerWidth, playerHeight)) # create a subsurface of the sprite
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png") # Save subsurface to a temporary file
temp_file.close()  # Close the temp file so pygame can access it
pygame.image.save(character_idle, temp_file.name) # save the subsurface to a temp file
sprite1 = lights_engine.load_texture(temp_file.name) # Load texture using the temporary file path
sprite1_pygame = pygame.image.load(temp_file.name) # load tempfile as a pygame image to get rect from
sprite1Rect = sprite1_pygame.get_rect() # creates the rect of sprite1

""" Load sprite from spritesheet """
def get_image(sheet, width, height):
    image = pygame.Surface((width, height)) # create a surface for the image
    image.set_colorkey((0, 0, 0))
    image.blit(sheet, (0, 0), (280, 22, 296, 36)) # specify the specific coordinats for the sprite
    return image

""" Update the Player Pos """
def updatePlayerPosVariable():
    global playerPos, playerX, playerY
    playerPos = [playerX, playerY] # update player pos
updatePlayerPosVariable() # update the play pos once on run to enable movement

""" Debug info """
def playerDebugInfo():
    global debug, playerPos
    if debug == True: # when debug is enabled
        # print(f"LastPos: {lastPlayerPos}") # print the players last recorded pos
        print(f"Pos: {playerPos}") # print player pos

""" Function to load rects from a CSV file """
def load_rects_from_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for y, row in enumerate(reader):
            for x, value in enumerate(row):
                # The values are read as strings, so we check for '0.0'
                if value.strip() == '0.0':
                    # Create a rect with a size of cellSize and position it
                    rect = pygame.Rect(x * cellSize, y * cellSize, cellSize, cellSize)
                    rects.append(rect)

""" Convert the list of Rects to a list of Hulls for the Lighting Engine """
def rects_to_multiple_hulls(rects):
    """
    Converts a list of pygame.Rect objects into a list of optimized hull lists,
    handling complex junctions and disconnected map structures via angle sorting.
    """
    if not rects:
        return []

    # --- Steps 1 & 2: Edge Mapping and Filtering ---
    edge_counts = defaultdict(int)
    for rect in rects:
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        segments = [ ((x, y), (x + w, y)), ((x + w, y), (x + w, y + h)),
                     ((x + w, y + h), (x, y + h)), ((x, y + h), (x, y)) ]
        for p1, p2 in segments:
            normalized_edge = tuple(sorted((p1, p2)))
            edge_counts[normalized_edge] += 1

    exterior_edges = set()
    for rect in rects:
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        segments = [ ((x, y), (x + w, y)), ((x + w, y), (x + w, y + h)),
                     ((x + w, y + h), (x, y + h)), ((x, y + h), (x, y)) ]
        
        for p1, p2 in segments:
            normalized_edge = tuple(sorted((p1, p2)))
            if edge_counts[normalized_edge] == 1:
                exterior_edges.add((p1, p2))

    # --- Step 3: Trace and Order Hulls (Now Robust with Angle Sorting) ---
    
    # Map points to their outgoing edges: {start_point: [end_point1, end_point2, ...]}
    outgoing_map = defaultdict(list)
    for p1, p2 in exterior_edges:
        outgoing_map[p1].append(p2)

    all_hulls = []

    # Keep tracing until all exterior edges have been consumed
    while outgoing_map:
        # Start a new hull trace with an arbitrary remaining edge's start point
        # Get the first key and its first outgoing point to set the starting segment
        start_point = next(iter(outgoing_map.keys()))
        current_point = start_point
        
        # Initialize the trace: (x_prev, y_prev) to help calculate the entry angle
        prev_point = (0, 0) # Placeholder, will be replaced by the point before 'start_point'
        
        # Find the first segment (start_point -> next_point) and use it to set prev_point
        # We need to find the point that *comes from* start_point, then trace from there.
        # This part requires a slight adjustment to ensure we enter the loop correctly.
        
        # Look at the *incoming* edges to 'start_point' to establish the starting direction
        incoming_point = None
        for p_start, p_end_list in outgoing_map.items():
            if start_point in p_end_list:
                incoming_point = p_start
                break
        
        if incoming_point is None:
             # If no incoming edge found, this must be an isolated single-segment hull (unlikely) or an error
             # For safety, just break and move to the next structure
             for p in outgoing_map[start_point]: # Consume all outgoing edges from this point
                 del outgoing_map[start_point]
             continue

        # Start the hull list. The order is crucial.
        current_hull = [list(incoming_point)] # The point before the official start of the segment
        current_point = incoming_point 
        next_point = start_point
        
        # Helper to calculate the angle of a segment (p1 to p2)
        def get_angle(p1, p2):
            return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

        # Main tracing loop
        while True:
            # Add the current point (where we just arrived)
            current_hull.append(list(next_point))
            
            # The edge we just arrived on is from (current_point -> next_point)
            # Its angle is the basis for our next turn.
            entry_angle = get_angle(current_point, next_point)
            
            # Update the point references for the next iteration
            current_point = next_point
            
            if current_point == start_point and len(current_hull) > 2:
                # Loop complete! We found our way back to the original start_point
                break 

            # Get all possible outgoing points from the new current_point
            possible_next_points = outgoing_map.get(current_point, [])
            
            if not possible_next_points:
                # This should not happen if the hull is closed and connected
                print(f"Error: Disconnected hull trace at {current_point} - No outgoing edges found.")
                break

            # --- Critical Fix: Angle Sorting ---
            
            # We want to choose the next edge that turns "left" relative to the entry_angle.
            # This is achieved by finding the edge with the smallest angular change (positive is a left turn)
            
            min_turn_angle = float('inf')
            best_next_point = None

            for p_next in possible_next_points:
                outgoing_angle = get_angle(current_point, p_next)
                
                # Calculate the difference (turn angle)
                turn_angle = outgoing_angle - entry_angle
                
                # Normalize the angle difference to the range (-pi, pi]
                # This ensures we correctly compare small turns vs large turns (left vs right)
                if turn_angle <= -math.pi:
                    turn_angle += 2 * math.pi
                elif turn_angle > math.pi:
                    turn_angle -= 2 * math.pi

                # We look for the largest *positive* turn (closest to 0 or a small left turn)
                # or, most robustly, we look for the outgoing edge that is immediately 
                # clockwise (smallest change in angle) relative to the reverse of the entry vector.
                
                # For a grid-based map, simply sorting the *outgoing* angles and selecting 
                # the one that is closest to -180 degrees relative to the entry angle
                # often works, but let's stick to the simplest working method for this constraint:
                
                # To trace the exterior, we look for the next edge that provides the 
                # **smallest counter-clockwise (left) turn** relative to the entry angle.
                # A robust grid-based approach is often easier: check directions (0, 90, 180, 270)
                # and prioritize the one that doesn't immediately enter an adjacent solid tile.
                
                # Since the angles are constrained (0, 90, 180, 270), we can just pick the first
                # available point that continues the line or turns a corner, assuming the edge 
                # filtering already handled the complex interior. We only need the correct 
                # direction when multiple options are left (e.g., at an end-cap).
                
                # Simple grid-based rule: find the edge that is closest to 
                # being straight-ahead or a smooth corner. We will prioritize the smallest 
                # angle change which is > 0 (left turn).
                
                if turn_angle > 1e-6: # Check for positive turn (left)
                    if turn_angle < min_turn_angle:
                        min_turn_angle = turn_angle
                        best_next_point = p_next
            
            # If a clear positive turn wasn't found (like at a straight line or an end)
            if best_next_point is None:
                # Default to the point that is "straightest" ahead (turn_angle near 0)
                # If no left turn is possible, take the straight-ahead path.
                min_turn_angle = float('inf')
                for p_next in possible_next_points:
                    outgoing_angle = get_angle(current_point, p_next)
                    turn_angle = outgoing_angle - entry_angle
                    if turn_angle <= -math.pi: turn_angle += 2 * math.pi
                    elif turn_angle > math.pi: turn_angle -= 2 * math.pi

                    if abs(turn_angle) < abs(min_turn_angle):
                        min_turn_angle = turn_angle
                        best_next_point = p_next

            if best_next_point is None:
                # Final fallback, just take the first one (shouldn't happen)
                best_next_point = possible_next_points[0]
            
            # Consume the edge and update the point
            next_point = best_next_point
            outgoing_map[current_point].remove(next_point)
            if not outgoing_map[current_point]:
                del outgoing_map[current_point]

        if current_hull:
            # The first point in current_hull is the incoming_point, which duplicates the end point
            # The start_point (index 1) is the true beginning of the loop
            all_hulls.append(current_hull[1:])

    return all_hulls

""" Randomly Load the Maze """
def randomMaze():
    global debug, numOfMazes
    mazeNum = random.randint(0,numOfMazes-1)
    randomMaze = f"{mazeDir}/maze_{mazeNum}.csv" # define a path for each new maze
    load_rects_from_csv(randomMaze) # load the maze
    if debug == True: print(f"Maze Number: {mazeNum}")
    
    return mazeNum
randomMaze()

""" Quit and Exit """
def quitExit():
    pygame.quit()
    if debug == True: print("Game Closed")
    exit()

# print(rects)

hulls = rects_to_multiple_hulls(rects)
print(hulls)

running = True # Variable to keep our game loop running

clock = pygame.time.Clock()
dt = 0 # Initialize dt to 0 for the first frame
def mainGameLoop():
    global debug, f2_pressed, is_won, args, windowDimentions, screen, bgColour, black, white, red, translucentBlack, featheredSize, currentFeatheredSize, featherdShadowMask, featherdShadowRect, playerHeight, playerWidth, playerX, playerY, speed, playerPos_Pause, running, clock, dt

    playerFall = True # does player fall with gravity

    sprite1Rect.x = playerX
    sprite1Rect.y = playerY

    if is_won == True:
        # reset player position
        playerX = 0
        playerY = 0
        is_won = False

    """ Game Loop """
    while running:
        # clock.tick(120) is now used to calculate the time passed
        # and returns the time in milliseconds
        dt = clock.tick(120) / 1000.0 # Calculate Delta Time in seconds (dt)
        
        # Calculate the actual distance moved this frame:
        # distance = speed (pixels/sec) * dt (seconds)
        distance_moved = speed * dt
        
        screen.fill(bgColour) # set background colour
        
        screen.blit(sprite1, (playerX, playerY)) # blit the player to the screen using the sprite specifies earlier and the player x & y values
        
        if debug == True: pygame.draw.rect(screen, red, sprite1Rect, 1) # Draw the hitbox (rect) of sprite1

        # Proccess events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            """ KEYDOWN Events """
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: # check if the escape key is pressed
                    playerPos_Pause = [playerX, playerY]
                    playerFall = False # make player stop falling with gravity
                    pauseMenu() # open pause menu
                if event.key == pygame.K_F2:
                    if not f2_pressed: # Check if the flag is False before running the code
                        debug = not debug # toggle debug flag
                        print(f"Debug: {debug}") # print the key debug state
                        f2_pressed = True # Set the flag to True so it won't run again until released

            """ KEYUP Events """
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_F2:
                    f2_pressed = False # Reset the flag when the key is released
                if args.gravity == 1 and playerFall == False:
                    if event.key == pygame.K_UP or pygame.K_w:
                        playerFall = True

        # Draw all the rects (walls) in the list
        for rect in rects:
            pygame.draw.rect(screen, black, rect)

        if featheredSize != currentFeatheredSize:
            currentFeatheredSize = featheredSize

            featherdShadowMask = generate_feathered_light_mask(shadowRadius, featheredSize)
            featherdShadowRect = featherdShadowMask.get_rect()

        """ Draw the shadow """
        if args.shadow == 1:
            shadowSurface.fill(black) # Fill the shadow Surface black for remove old shadows
            # Note: Using playerX and playerY directly here is fine as they are float-based
            pygame.draw.circle(shadowSurface, translucentBlack, ((int(playerX+(playerHeight/2))), (int(playerY+(playerWidth/2)))), shadowRadius) # Draw the shadow onto the shadowSurface with the transparent colour and centred to sprite1
            """ Draw the feathered shadow effect """
            featherdShadowRect.center = ((int(playerX+(playerHeight/2))), (int(playerY+(playerWidth/2))))
            shadowSurface.blit(featherdShadowMask, featherdShadowRect, special_flags=pygame.BLEND_RGBA_MIN)
            screen.blit(shadowSurface, (0, 0)) # Blit the shadowSurface the the top left of the screen


        pygame.display.flip()
        
        """ Player Movement with Collisions """
        keys = pygame.key.get_pressed()
        
        # Store the player's current position before attempting to move
        # Note: playerX/Y are now floats, and sprite1Rect.x/y are integers
        oldX = playerX
        oldY = playerY
        oldRectX = sprite1Rect.x
        oldRectY = sprite1Rect.y

        # --- Horizontal Movement ---
        if (keys[pygame.K_LEFT] or keys[pygame.K_a] or keys[pygame.K_j]):
            # Move the float-based position by distance_moved
            playerX -= distance_moved 
            # Update the integer-based rect position by converting the float to an integer
            sprite1Rect.x = int(playerX)
            updatePlayerPosVariable() 
            playerDebugInfo()
        elif (keys[pygame.K_RIGHT] or keys[pygame.K_d] or keys[pygame.K_l]):
            # Move the float-based position by distance_moved
            playerX += distance_moved 
            # Update the integer-based rect position by converting the float to an integer
            sprite1Rect.x = int(playerX)
            updatePlayerPosVariable() 
            playerDebugInfo() 

        # Check for horizontal boundary collision after the attempted move (using the rect's int position)
        if sprite1Rect.x < 0 or sprite1Rect.x > windowDimentions[0] - playerWidth:
            # Revert the x-position if boundary is hit
            playerX = oldX
            sprite1Rect.x = oldRectX
            updatePlayerPosVariable()
        else:
            if args.wall_collisions == 1:
                # Check for wall collisions only if boundary wasn't hit
                verticalCollision = False
                for rect in rects:
                    if sprite1Rect.colliderect(rect):
                        verticalCollision = True
                        break
                
                if verticalCollision:
                    # Revert the x-position if a wall collision occurred
                    playerX = oldX
                    sprite1Rect.x = oldRectX
                    updatePlayerPosVariable()
                    if debug: print("Vertical wall collision detected!")

        # --- Gravity ---
        if playerFall == True and args.gravity == 1:
            playerY += distance_moved/2
            sprite1Rect.y = int(playerY)
   
        # --- Vertical Movement ---
        if (keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_i]):
            playerFall = False
            # Move the float-based position by distance_moved
            playerY -= distance_moved 
            # Update the integer-based rect position by converting the float to an integer
            sprite1Rect.y = int(playerY)
            updatePlayerPosVariable() 
            playerDebugInfo()
        elif (keys[pygame.K_DOWN] or keys[pygame.K_s] or keys[pygame.K_k]):
            # Move the float-based position by distance_moved
            playerY += distance_moved 
            # Update the integer-based rect position by converting the float to an integer
            sprite1Rect.y = int(playerY) 
            updatePlayerPosVariable() 
            playerDebugInfo() 

        # Check for vertical boundary collision after the attempted move (using the rect's int position)
        if sprite1Rect.y < 0 or sprite1Rect.y > windowDimentions[1] - playerHeight:
            # Revert the y-position if boundary is hit
            playerY = oldY
            sprite1Rect.y = oldRectY
            updatePlayerPosVariable()
        else:
            if args.wall_collisions == 1:
                # Check for wall collisions only if boundary wasn't hit
                horizontalCollision = False
                for rect in rects:
                    if sprite1Rect.colliderect(rect):
                        horizontalCollision = True
                        break

                if horizontalCollision:
                    # Revert the y-position if a wall collision occurred
                    playerY = oldY
                    sprite1Rect.y = oldRectY
                    updatePlayerPosVariable()
                    if debug: print("Horizontal wall collision detected!")
        
        # --- Finished Maze ---
        if sprite1Rect.y > 920 and sprite1Rect.x > 920:
            is_won = True
            screen.fill(black) # clear the screen
            winScreen() # open the home menu
            break

def homeMenu():
    global debug, args, screen, bgColour, black, white, red, translucentBlack, playerX, playerY, running

    """ Load Buttons """
    startButton = Button(
        image_path="assets/buttons/Menu Buttons/Large Buttons/Large Buttons/Start Button.png",
        position=(231, 210),
        scale=0.8,
        hover_image_path="assets/buttons/Menu Buttons/Large Buttons/Coloured Large Buttons/Start col_Button.png"
    )
    settingsButton = Button(
        image_path="assets/buttons/Menu Buttons/Large Buttons/Large Buttons/Settings Button.png",
        position=(231, 390),
        scale=0.8,
        hover_image_path="assets/buttons/Menu Buttons/Large Buttons/Coloured Large Buttons/Settings col_Button.png"
    )
    quitButton = Button(
        image_path="assets/buttons/Menu Buttons/Large Buttons/Large Buttons/Quit Button.png",
        position=(231, 703),
        scale=0.8,
        hover_image_path="assets/buttons/Menu Buttons/Large Buttons/Coloured Large Buttons/Quit col_Button.png"
    )

    while running:
        screen.fill(bgColour) # set background colour

        """ Render "Blind Maze" to top of screen """
        gameName = KightWarrior_Font_Large.render("Blind Maze", False, white)
        screen.blit(gameName, (20, 15)) 

        """ Draw Buttons """
        startButton.draw(screen)
        settingsButton.draw(screen)
        quitButton.draw(screen)

        pygame.display.flip()

        # Proccess events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            """ Button Presses """
            if startButton.is_pressed():
                playerX = 0 # reset playerX
                playerY = 0 # reset playerY
                screen.fill(black) # clear the screen
                mainGameLoop() # run the game loop
 
            if settingsButton.is_pressed():
                screen.fill(black) # clear the screen
                settingsMenu("home") # open the settings menu

            if quitButton.is_pressed():
                running = False

            """ Button Hover """
            startButton.is_hover()
            settingsButton.is_hover()
            quitButton.is_hover()

def pauseMenu():
    global debug, args, screen, bgColour, black, white, red, translucentBlack, rects, playerX, playerY, running

    """ Load Buttons """
    resumeButton = Button(
        image_path="assets/buttons/Menu Buttons/Large Buttons/Large Buttons/Resume Button.png",
        position=(25, 260),
        scale=0.7,
        hover_image_path="assets/buttons/Menu Buttons/Large Buttons/Coloured Large Buttons/Resume col_Button.png"
    )
    newGameButton = Button(
        image_path="assets/buttons/Menu Buttons/Large Buttons/Large Buttons/New Game Button.png",
        position=(498, 260),
        scale=0.7,
        hover_image_path="assets/buttons/Menu Buttons/Large Buttons/Coloured Large Buttons/New Game col_Button.png"
    )
    settingsButton = Button(
        image_path="assets/buttons/Menu Buttons/Large Buttons/Large Buttons/Settings Button.png",
        position=(25, 590),
        scale=0.5,
        hover_image_path="assets/buttons/Menu Buttons/Large Buttons/Coloured Large Buttons/Settings col_Button.png"
    )
    quitButton = Button(
        image_path="assets/buttons/Menu Buttons/Large Buttons/Large Buttons/Quit Button.png",
        position=(25, 805),
        scale=0.5,
        hover_image_path="assets/buttons/Menu Buttons/Large Buttons/Coloured Large Buttons/Quit col_Button.png"
    )

    while running:
        screen.fill(bgColour)

        """ Render "Blind Maze" to top of screen """
        gameName = KightWarrior_Font_Large.render("Blind Maze", False, white)
        screen.blit(gameName, (20, 15))

        """ Draw Buttons """
        resumeButton.draw(screen)
        newGameButton.draw(screen)
        settingsButton.draw(screen)
        quitButton.draw(screen)

        pygame.display.update()

        # Proccess events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            """ Button Presses """
            if resumeButton.is_pressed():
                screen.fill(black) # clear the screen
                mainGameLoop() # run the game loop

            if newGameButton.is_pressed():
                playerX = 0 # reset playerX
                playerY = 0 # reset playerY
                rects = [] # clear the rects (walls)
                randomMaze() # select a new maze
                screen.fill(black) # clear the screen
                mainGameLoop() # run the game loop

            if settingsButton.is_pressed():
                screen.fill(black) # clear the screen
                settingsMenu("pause") # open the settings menu

            if quitButton.is_pressed():
                running = False

            """ Button Hover """
            resumeButton.is_hover()
            newGameButton.is_hover()
            settingsButton.is_hover()
            quitButton.is_hover()

def settingsMenu(lastMenu):
    global debug, args, screen, bgColour, black, white, red, translucentBlack, shadowRadius, featheredSize, running

    """ Load Buttons """
    if lastMenu == "home":
        playButton = Button(
            image_path="assets/buttons/Menu Buttons/Large Buttons/Large Buttons/Start Button.png",
            position=(25, 140),
            scale=0.5,
            hover_image_path="assets/buttons/Menu Buttons/Large Buttons/Coloured Large Buttons/Start col_Button.png"
        )
    elif lastMenu == "pause":
        playButton = Button(
            image_path="assets/buttons/Menu Buttons/Large Buttons/Large Buttons/Resume Button.png",
            position=(25, 140),
            scale=0.5,
            hover_image_path="assets/buttons/Menu Buttons/Large Buttons/Coloured Large Buttons/Resume col_Button.png"
        )
    quitButton = Button(
        image_path="assets/buttons/Menu Buttons/Large Buttons/Large Buttons/Quit Button.png",
        position=(25, 805),
        scale=0.5,
        hover_image_path="assets/buttons/Menu Buttons/Large Buttons/Coloured Large Buttons/Quit col_Button.png"
    )
    backButton = Button(
        image_path="assets/buttons/Menu Buttons/Square Buttons/Square Buttons/Back Square Button.png",
        position=(330, 260),
        scale=0.3,
        hover_image_path="assets/buttons/Menu Buttons/Square Buttons/Coloured Square Buttons/Back col_Square Button.png"
    )
    nextButton = Button(
        image_path="assets/buttons/Menu Buttons/Square Buttons/Square Buttons/Next Square Button.png",
        position=(473, 260),
        scale=0.3,
        hover_image_path="assets/buttons/Menu Buttons/Square Buttons/Coloured Square Buttons/Next col_Square Button.png"
    )
    tickButtonPos = (225, 350)
    tickButton = Button(
        image_path="assets/buttons/Menu Buttons/Square Buttons/Square Buttons/V Square Button.png",
        position=tickButtonPos,
        scale=0.3,
        hover_image_path="assets/buttons/Menu Buttons/Square Buttons/Coloured Square Buttons/V col_Square Button.png"
    )


    tickButton_State = "V"

    while running:
        if debug == True and tickButton_State == "X":
            tickButton = Button(
                image_path="assets/buttons/Menu Buttons/Square Buttons/Square Buttons/V Square Button.png",
                position=tickButtonPos,
                scale=0.3,
                hover_image_path="assets/buttons/Menu Buttons/Square Buttons/Coloured Square Buttons/V col_Square Button.png"
            )
            tickButton_State = "V"
        elif debug == False and tickButton_State == "V":
            tickButton = Button(
                image_path="assets/buttons/Menu Buttons/Square Buttons/Square Buttons/X Square Button.png",
                position=tickButtonPos,
                scale=0.3,
                hover_image_path="assets/buttons/Menu Buttons/Square Buttons/Coloured Square Buttons/X col_Square Button.png"
            )
            tickButton_State = "X"

        screen.fill(bgColour) # set background colour

        """ Render "Blind Maze" to top of screen """
        gameName = KightWarrior_Font_Large.render("Blind Maze", False, white)
        screen.blit(gameName, (20, 15)) 

        """ Options Text """
        player_fall_on = MinecraftEvenings_Font_Medium.render("Debug:", False, white)
        screen.blit(player_fall_on, (25, 355))

        shadow_size = MinecraftEvenings_Font_Medium.render("Light Size:", False, white)
        screen.blit(shadow_size, (25, 265))

        shadow_size_value = MinecraftEvenings_Font_Medium.render(str(shadowRadius), False, white)
        screen.blit(shadow_size_value, (400, 264))

        """ Draw Buttons """
        playButton.draw(screen)
        tickButton.draw(screen)
        backButton.draw(screen)
        nextButton.draw(screen)
        quitButton.draw(screen)

        pygame.display.flip()

        # Proccess events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            """ Button Presses """
            if playButton.is_pressed():
                screen.fill(black) # clear the screen
                mainGameLoop() # run the game loop

            if tickButton.is_pressed():
                debug = not debug # toggle debug flag
                print(f"Debug: {debug}") # print the key debug state
        
            if backButton.is_pressed():
                if shadowRadius > 20:
                    shadowRadius -= 10
                    featheredSize -= 15
                if debug == True:
                    print(f"Shadow Radius set to: {shadowRadius}")
                    print(f"Featherd Size set to: {featheredSize}")
            if nextButton.is_pressed():
                if shadowRadius < 90:
                    shadowRadius += 10
                    featheredSize += 15
                if debug == True:
                    print(f"Shadow Radius set to: {shadowRadius}")
                    print(f"Featherd Size set to: {featheredSize}")

            if quitButton.is_pressed():
                running = False

            """ Button Hover """
            playButton.is_hover()
            quitButton.is_hover()
            backButton.is_hover()
            nextButton.is_hover()
            tickButton.is_hover()

def winScreen():
    global debug, args, screen, bgColour, black, white, red, translucentBlack, rects, playerX, playerY, running

    """ Load Buttons """
    newGameButton = Button(
        image_path="assets/buttons/Menu Buttons/Large Buttons/Large Buttons/New Game Button.png",
        position=(231, 425),
        scale=0.8,
        hover_image_path="assets/buttons/Menu Buttons/Large Buttons/Coloured Large Buttons/New Game col_Button.png"
    )
    quitButton = Button(
        image_path="assets/buttons/Menu Buttons/Large Buttons/Large Buttons/Quit Button.png",
        position=(321, 805),
        scale=0.5,
        hover_image_path="assets/buttons/Menu Buttons/Large Buttons/Coloured Large Buttons/Quit col_Button.png"
    )

    while running:
        screen.fill(bgColour)

        """ Render "Blind Maze" to top of screen """
        youWon = KightWarrior_Font_XLarge.render("You Won!!!", False, white)
        screen.blit(youWon, (234, 25))

        """ Draw Buttons """
        newGameButton.draw(screen)
        quitButton.draw(screen)

        pygame.display.update()

        # Proccess events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            """ Button Presses """
            if newGameButton.is_pressed():
                playerX = 0 # reset playerX
                playerY = 0 # reset playerY
                rects = [] # clear the rects (walls)
                randomMaze() # select a new maze
                screen.fill(black) # clear the screen
                mainGameLoop() # run the game loop

            if quitButton.is_pressed():
                running = False

            """ Button Hover """
            newGameButton.is_hover()
            quitButton.is_hover()

homeMenu()