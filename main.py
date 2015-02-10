#from liblas import file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import math

from vispy import app
from vispy import gloo
from vispy.util.transforms import perspective, translate, xrotate, yrotate
from vispy.util.transforms import zrotate
from scipy.spatial import Delaunay

triangles = []
height = 8.0
valmin = 0;
valmax = 0;

#//v_color = vec4(0.0, a_position[2] * a_position[2] / (u_height * u_height * u_height), 0.1, 1.0);

vertex = """
uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;

attribute vec3 water;
attribute vec3 a_position;

varying vec3 v_position;
varying vec3 v_normal;
//varying vec4 v_color;


void main (void) {
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0); 
    //v_color = vec4(0.0, a_position[2] * a_position[2] / (u_height * u_height * u_height), 0.1, 1.0);
    //if(a_position[2]<water.z){
    //    v_color = vec4(0.0,0.0,1.0,0.5);
    //}
    //else if((mod(a_position[2],1.0) > 0.0) && (mod(a_position[2],1.0) < 0.2)){
    //    v_color = vec4(0.1,0.2,0.1,1.0);
    //}
    //else{
    //    v_color = vec4(0.0,1.0,0.0,1.0);
    //}
    v_position = a_position;
}
"""

fragment = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_normal;

uniform vec3 u_light_intensity;
uniform vec3 u_light_position;

varying vec3 v_position;
//varying vec3 v_normal;
//varying vec4 v_color;

const vec4 dark = vec4(0.1,0.2,0.1,1.0);
const vec4 green = vec4(0.0,1.0,0.0,1.0);

void main(){
    vec3 v_normal = normalize(cross(dFdy(v_position), dFdx(v_position))); // N is the world normal

    // Calculate normal in world coordinates
    vec3 normal = normalize(u_normal * vec4(v_normal,1.0)).xyz;

    // Calculate the location of this fragment (pixel) in world coordinates
    vec3 position = vec3(u_view*u_model * vec4(v_position, 1));

    // Calculate the vector from this pixels surface to the light source
    vec3 surfaceToLight = u_light_position - position;

    // Calculate the cosine of the angle of incidence (brightness)
    float brightness = dot(normal, surfaceToLight) /
                      (length(surfaceToLight) * length(normal));
    brightness = max(min(brightness,1.0),0.0);

    // Calculate final color of the pixel, based on:
    // 1. The angle of incidence: brightness
    // 2. The color/intensities of the light: light.intensities
    // 3. The texture and texture coord: texture(tex, fragTexCoord)

    float z = v_position.z;
    vec4 v_color = vec4(0.0,0.0,0.0,0.0);
    vec4 red = vec4(1.0,0.0,0.0,1.0);
    vec4 green = vec4(0.0,1.0,0.0,1.0);
    vec4 blue = vec4(0.0,0.0,1.0,1.0);
    vec4 orange = vec4(1.0,0.46,0.0,1.0);
    vec4 ice = vec4(0.0,1.0,1.0,1.0);
    vec4 black = vec4(0.0,0.0,0.0,1.0);
        
    //if(mod(z,1.0) > 0.95 && mod(z,1.0) < 0.05) {
    //    v_color = black;//mix(v_color, black, smoothstep(0.0, 0.05, mod(z,1.0))-smoothstep(0.95, 1.0, mod(z,1.0)));
    //}
    //else {
    v_color = mix(blue, blue, smoothstep(0, 68, z));
    v_color = mix(v_color, ice, smoothstep(68, 74, z));
    v_color = mix(v_color, green, smoothstep(74, 80, z));
    v_color = mix(v_color, orange, smoothstep(86, 92, z));
    v_color = mix(v_color, red, smoothstep(92, 100, z));
    //}
    v_color = mix(black, v_color, smoothstep(0.0, 0.05, mod(z,1.0))-smoothstep(0.95, 1.0, mod(z,1.0)));

    gl_FragColor = v_color * brightness * vec4(u_light_intensity, 1);
}
"""

class Canvas(app.Canvas):
    lightx = -625
    lighty = -625
    lightz = -2000
    water = 60

    def __init__(self):
        app.Canvas.__init__(self, keys='interactive')

        self.program = gloo.Program(vertex, fragment)

        


        
        #Sets the view to an appropriate position over the terrain
        #self.default_view = np.array([[0.8, 0.2, -0.48, 0],
        #                             [-0.5, 0.3, -0.78, 0],
        #                             [-0.01, 0.9, -0.3, 0],
        #                             [-4.5, -21.5, -7.4, 1]],
        #                             dtype=np.float32)

        #self.default_view = np.array([[0.8, 0.2, -0.48, 0],
        #                             [-0.5, 0.3, -0.78, 0],
        #                             [-0.01, 0.9, -0.3, 0],
        #                             [-4.5, -21.5, -7.4, 1]],
        #                             dtype=np.float32)



        #print self.default_view

        self.default_view = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]],dtype=np.float32)
        self.view = self.default_view
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.translate = [0, 0, 0]
        self.rotate = [0, 0, 0]

        #self.program["u_light_position"] = -40, -40, -40


        self.program["u_light_position"] = self.lightx,self.lighty,self.lightz
        self.program["u_light_intensity"] = 1, 1, 1

        #self.program['u_height'] = height
        self.program['u_model'] = self.model

        translate(self.view, -625, -625, -1000)
        zrotate(self.view, 90)
        self.program['u_view'] = self.view

        

        self.program['u_normal'] = np.array(np.matrix(np.dot(self.view, self.model)).I.T)

        self.program['a_position'] = gloo.VertexBuffer(triangles)
        self.program['water'] = 0,0,self.water
        self.update()

#uniform vec3 u_light_intensity;
#uniform vec3 u_light_position;

#varying vec3 a_position;
#varying vec3 v_normal;
#varying vec4 v_color;

        self.program['a_position'] = gloo.VertexBuffer(triangles)

    def on_initialize(self, event):
        gloo.set_state(clear_color='black', depth_test=True)

    def on_key_press(self, event):
        """Controls -
        a(A) - move left
        d(D) - move right
        w(W) - move up
        s(S) - move down
        x/X - rotate about x-axis cw/anti-cw
        y/Y - rotate about y-axis cw/anti-cw
        z/Z - rotate about z-axis cw/anti-cw
        space - reset view
        p(P) - print current view
        i(I) - zoom in
        o(O) - zoom out
        """
        self.translate = [0, 0, 0]
        self.rotate = [0, 0, 0]

        if(event.text == 'p' or event.text == 'P'):
            print(self.view)
        elif(event.text == 'd' or event.text == 'D'):
            self.translate[0] = 30.
        elif(event.text == 'a' or event.text == 'A'):
            self.translate[0] = -30.
        elif(event.text == 'w' or event.text == 'W'):
            self.translate[1] = 30.
        elif(event.text == 's' or event.text == 'S'):
            self.translate[1] = -30.
        elif(event.text == 'o' or event.text == 'O'):
            self.translate[2] = 30.
        elif(event.text == 'i' or event.text == 'I'):
            self.translate[2] = -30.
        elif(event.text == 'x'):
            self.rotate = [1, 0, 0]
        elif(event.text == 'X'):
            self.rotate = [-1, 0, 0]
        elif(event.text == 'y'):
            self.rotate = [0, 1, 0]
        elif(event.text == 'Y'):
            self.rotate = [0, -1, 0]
        elif(event.text == 'z'):
            self.rotate = [0, 0, 1]
        elif(event.text == 'Z'):
            self.rotate = [0, 0, -1]
        elif(event.text == 'h'):
            self.rotate = [0, 0, 0]
        elif(event.text == 'j'):
            self.rotate = [90, 0, 0]
        elif(event.text == 'k'):
            self.rotate = [180, 0, 0]
        elif(event.text == 'l'):
            self.rotate = [270, 0, 0]            
        elif(event.text == ' '):
            self.view = self.default_view
        elif(event.text == '6'):
            self.lightx += 100.
        elif(event.text == '4'):
            self.lightx += -100.
        elif(event.text == '8'):
            self.lighty += 100.
        elif(event.text == '5'):
            self.lighty += -100.
        elif(event.text == '7'):
            self.lightz += 100.
        elif(event.text == '1'):
            self.lightz += -100.
        elif(event.text == '9'):
            self.water += .2
            print self.water
        elif(event.text == '3'):
            self.water += -.2
            print self.water

        translate(self.view, -self.translate[0], -self.translate[1],
                  -self.translate[2])
        xrotate(self.view, self.rotate[0])
        yrotate(self.view, self.rotate[1])
        zrotate(self.view, self.rotate[2])

        self.program["u_light_position"] = self.lightx,self.lighty,self.lightz
        print self.program["u_light_position"] 
        self.program['u_view'] = self.view
        self.program['water'] = 0,0,self.water
        self.update()

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)
        self.projection = perspective(60.0, width / float(height), 1.0, 2000.0)
        self.program['u_projection'] = self.projection

    def on_draw(self, event):
        # Clear
        gloo.clear(color=True, depth=True)
        # Draw
        self.program.draw('triangles')


# Create empty numpy array
#PointsXYZIC = np.empty(shape=(num_points, 5))
# Load all LAS points into numpy array


def generate_points():
    global triangles,valmin,valmax
    #f = file.File('09P001_67250_5950_25.las',mode='r')
    f = open('67250_5950_25.asc',mode='r')

    ncols = int(f.readline().split()[1]);
    nrows = int(f.readline().split()[1]);
    xllcenter = float(f.readline().split()[1]);
    yllcenter = float(f.readline().split()[1]);
    cellsize = float(f.readline().split()[1]);
    nodata_value = int(f.readline().split()[1]);

    print ncols
    print nrows
    print xllcenter
    print yllcenter
    print cellsize
    print nodata_value

    #num_points = 50000
    

    #x = np.zeros([ncols*nrows])
    #y = np.zeros([ncols*nrows])
    #z = np.zeros([ncols*nrows])
    row = 0
    col = 0
    index = 0
    arr = np.zeros((ncols*nrows,3));
    arr = arr.astype(np.float32)
    valmax = 0;
    valmin = 0;
    for line in f:
        valarr = line.split()
        for val in valarr:
            row = index/ncols
            col = index%ncols

            #x[index] = col
            #y[index] = row
            #z[index] = /1000.0;
            #arr = np.append(arr,[row,col,float(val)],axis=0);
            #val = (float(val)/10.0)-6.664
            #val *= 2
            val = (float(val))
            #print val/7.48
            arr[index] = [row,col,val]
            #if val > valmax:
            #    valmax = val

            #if index == 0:
            #    valmin = val

            #if val < valmin:
            #    valmin = val
            index += 1
            #col = col + 1;
        #row = row + 1;
        
        #if idx > (num_points-1):
        #    break
    #return
    #    x[idx]=(p.x-595180.09)/1564.72#-595000.0)
    #    y[idx]=(p.y-6725000.0)/2499.97#-6725000.0)
    #    z[idx]=(p.z-68.52)/35.71#-66.61)
    #arr = np.resize(arr, (ncols*nrows, 3))
    print valmin
    print valmax
    tri = Delaunay(np.delete(arr,2,1))
    triangles = arr[tri.simplices]
    triangles = np.vstack(triangles)
    #triangles = np.vstack(triangles)

    #linspace = np.linspace(0, 1, num_points)


    #program['a_position'] = np.c_[
    #    x,
    #    y,
    #    z]




generate_points()

if __name__ == '__main__':
    c = Canvas()
    c.show()
    app.run()


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(x, y, h, c=h, marker='+', s=200, alpha=.1, cmap='PRGn')
#ax.plot_surface(x, y, h)

#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')

#plt.show()
