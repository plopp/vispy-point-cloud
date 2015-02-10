# -*- coding: utf-8 -*-
# Author: Marcus Kempe
#
# Example data file can be downloaded from: http://www.lantmateriet.se/globalassets/kartor-och-geografisk-information/download/demo_grid2_5x5.zip
# Look for the .asc-file.
#
# Parts of this file is based on the following vispy gallery example: 
# 
# http://vispy.org/examples/demo/gloo/terrain.html
# 
# Any material contained in this document that can be directly backtracked to this example is
# licensed under that documents own license, everything else is unlicensed as follows: 
#
# This is free and unencumbered software released into the public domain.

# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.

# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# For more information, please refer to <http://unlicense.org/>

import numpy as np
import math
from vispy import app
from vispy import gloo
from vispy.util.transforms import perspective, translate, xrotate, yrotate, zrotate
from scipy.spatial import Delaunay

triangles = []

vertex = """
uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;

attribute vec3 a_position;

varying vec3 v_position;
varying vec3 v_normal;

void main (void) {
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0); 
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
    float brightness = dot(normal, surfaceToLight) / (length(surfaceToLight) * length(normal));
    brightness = max(min(brightness,1.0),0.0);

    float z = v_position.z;
    vec4 v_color = vec4(0.0,0.0,0.0,0.0);
    vec4 red = vec4(1.0,0.0,0.0,1.0);
    vec4 green = vec4(0.0,1.0,0.0,1.0);
    vec4 blue = vec4(0.0,0.0,1.0,1.0);
    vec4 orange = vec4(1.0,0.46,0.0,1.0);
    vec4 ice = vec4(0.0,1.0,1.0,1.0);
    vec4 black = vec4(0.0,0.0,0.0,1.0);

    v_color = mix(blue, blue, smoothstep(0, 68, z));
    v_color = mix(v_color, ice, smoothstep(68, 74, z));
    v_color = mix(v_color, green, smoothstep(74, 80, z));
    v_color = mix(v_color, orange, smoothstep(86, 92, z));
    v_color = mix(v_color, red, smoothstep(92, 100, z));

    //Plot contour lines with 1m equidistance
    v_color = mix(black, v_color, smoothstep(0.0, 0.05, mod(z,1.0))-smoothstep(0.95, 1.0, mod(z,1.0)));

    gl_FragColor = v_color * brightness * vec4(u_light_intensity, 1);
}
"""

class Canvas(app.Canvas):
    lightx = -625
    lighty = -625
    lightz = -2000

    def __init__(self):
        app.Canvas.__init__(self, keys='interactive')
        self.program = gloo.Program(vertex, fragment)
        self.default_view = np.eye(4, dtype=np.float32)
        self.view = self.default_view
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)
        self.translate = [0, 0, 0]
        self.rotate = [0, 0, 0]
        self.program["u_light_position"] = self.lightx,self.lighty,self.lightz
        self.program["u_light_intensity"] = 1, 1, 1
        self.program['u_model'] = self.model
        translate(self.view, -625, -625, -1000)
        zrotate(self.view, 90)
        self.program['u_view'] = self.view
        self.program['u_normal'] = np.array(np.matrix(np.dot(self.view, self.model)).I.T)
        self.program['a_position'] = gloo.VertexBuffer(triangles)
        self.update()
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
        6 - Move light right
        4 - Move light left
        8 - Move light up
        5 - Move light down
        7 - Move light up (depthwise)
        1 - Move light down (depthwise)
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

        translate(self.view, -self.translate[0], -self.translate[1],
                  -self.translate[2])
        xrotate(self.view, self.rotate[0])
        yrotate(self.view, self.rotate[1])
        zrotate(self.view, self.rotate[2])

        self.program["u_light_position"] = self.lightx,self.lighty,self.lightz
        self.program['u_view'] = self.view
        self.update()

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)
        self.projection = perspective(60.0, width / float(height), 1.0, 2000.0)
        self.program['u_projection'] = self.projection

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.program.draw('triangles')


def generate_points():
    global triangles
    f = open('67250_5950_25.asc',mode='r')

    ncols = int(f.readline().split()[1]);
    nrows = int(f.readline().split()[1]);
    xllcenter = float(f.readline().split()[1]);
    yllcenter = float(f.readline().split()[1]);
    cellsize = float(f.readline().split()[1]);
    nodata_value = int(f.readline().split()[1]);

    print "Columns in data file: ",ncols
    print "Rows in data file: ",nrows
    print "Coordinate X-wise center (SWEREF99): ",xllcenter
    print "Coordinate Y-wise center (SWEREF99): ",yllcenter
    print "Cell size in meters: ",cellsize
    print "Value if no data available in point: ",nodata_value

    row = 0
    col = 0
    index = 0
    arr = np.zeros((ncols*nrows,3));
    arr = arr.astype(np.float32)
    for line in f:
        valarr = line.split()
        for val in valarr:
            row = index/ncols
            col = index%ncols
            val = (float(val))
            arr[index] = [row,col,val]
            index += 1

    #Delaunay triangulation of laser points
    tri = Delaunay(np.delete(arr,2,1))
    triangles = arr[tri.simplices]
    triangles = np.vstack(triangles)

generate_points()

if __name__ == '__main__':
    c = Canvas()
    c.show()
    app.run()