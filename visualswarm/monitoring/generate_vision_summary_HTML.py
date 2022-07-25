import webbrowser
from visualswarm.contrib.puppetmaster import HOSTS, WEBCAM_HOSTS

filename = 'vision_summary.html'
f = open(filename, 'w')

message = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<title>VSWRM - Vision Summary</title>
</head>
<body>
<script>
function AutoChangeImage(){{
    document.getElementById('image_r1').src="http://{HOSTS["Robot1"]}:8000/"+ Date.now() +"stream.mjpg";
    document.getElementById('image_r2').src="http://{HOSTS["Robot2"]}:8000/"+ Date.now() +"stream.mjpg";
    document.getElementById('image_r3').src="http://{HOSTS["Robot3"]}:8000/"+ Date.now() +"stream.mjpg";
    document.getElementById('image_r4').src="http://{HOSTS["Robot4"]}:8000/"+ Date.now() +"stream.mjpg";
    document.getElementById('image_r5').src="http://{HOSTS["Robot5"]}:8000/"+ Date.now() +"stream.mjpg";
}}
window.onload = function () {{
    setInterval(AutoChangeImage, 1000);
}};
</script>
<table class="tg" width="100%">
<tbody>
<tr style="background-color: #999999;">
<th>
<h1>Robot Vision</h1>
</th>
<th>
<h1>Top View</h1>
</th>
</tr>
<tr>
<th class="tg-0lax" width="100%">
<table class="tg" width="100%">
<thead>
<tr>
<th class="tg-0lax"><center>
<h1>Robot1</h1>
</center><center><img id="image_r1" width="100%" height="100%" /></center></th>
<th class="tg-0lax"><center>
<h1>Robot2</h1>
</center><center><img id="image_r2" width="100%" height="100%" /></center></th>
<th class="tg-0lax"><center>
<h1>Robot3</h1>
</center><center><img id="image_r3" width="100%" height="100%" /></center></th>
</tr>
</thead>
<tbody>
<tr>
<td class="tg-0lax"><center>
<h1>Robot4</h1>
</center><center><img id="image_r4" width="100%" height="100%" /></center></td>
<td class="tg-0lax"><center>
<h1>Robot5</h1>
</center><center><img id="image_r5" width="100%" height="100%" /></center></td>
<td class="tg-0lax">&nbsp;</td>
</tr>
</tbody>
</table>
</th>
<th class="tg-0lax" width="100%"><center>
<h1><img id="image_c1" src="http://{WEBCAM_HOSTS["Birdseye Cam"]}:8000/stream.mjpg" width="600" height="600"></h1>
</center></th>
</tr>
<tr height="40"></tr>
<tr>
<td colspan="2"><center>
<div>
    <img src="https://www.scienceofintelligence.de/wp-content/uploads/2018/11/scioi_logo@2x.png" alt="logo" height="100"/>
    <img src="https://www.projekte.hu-berlin.de/de/rueg-p3/logo-ordner/logo-hu/@@download/Image/HU-Logo.jpg" alt="logo" height="100"/>
    <img src="https://www.trr154.fau.de/wp-content/themes/TRR_21102020/images/LogoTUBerlin.jpg" alt="logo" height="90"/>
</div>
SCIoI P34 VSWRM</center>
</td>
</tr>
</tbody>
</table>
</body>
</html>
"""

f.write(message)
f.close()

#Change path to reflect file location
webbrowser.open_new_tab(filename)