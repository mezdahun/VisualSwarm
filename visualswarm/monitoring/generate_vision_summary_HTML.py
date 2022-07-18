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
    document.getElementById('image_c1').src="http://{WEBCAM_HOSTS["Birdseye Cam"]}:8000/"+ Date.now() +"stream.mjpg";
}}
window.onload = function () {{
    setInterval(AutoChangeImage, 1000);
}};
</script>
<table class="tg" height="100%" width="100%">
  <tr height="100%">
    <th class="tg-0lax" width="100%">
        <center><h1>Robot Vision</h1></center>        
        <table class="tg" height="100%" width="100%">
            <thead>
              <tr height="50%">
                <th class="tg-0lax">
                    <center><h1>Robot1</h1></center>
                    <center><img id="image_r1" src="" width="100%" height="100%"></center>
                </th>
                <th class="tg-0lax">
                        <center><h1>Robot2</h1></center>
                        <center><img id="image_r2" src="" width="100%" height="100%"></center></th>
                <th class="tg-0lax">
                        <center><h1>Robot3</h1></center>
                        <center><img id="image_r3" src="" width="100%" height="100%"></center></th>
              </tr>
            </thead>
            <tbody>
              <tr height="50%">
                <td class="tg-0lax">
                        <center><h1>Robot4</h1></center>
                        <center><img  id="image_r4" src="" width="100%" height="100%"></center>
                </td>
                <td class="tg-0lax">
                        <center><h1>Robot5</h1></center>
                        <center><img  id="image_r5" src="" width="100%" height="100%"></center>
                </td>
                <td class="tg-0lax"></td>
              </tr>
            </tbody>
            </table>
        
    </th>
    <th class="tg-0lax" width="100%">
        <center><h1>Birdseye View</h1></center>
        <center><img  id="image_c1" src="" width="600" height="600"></center>
    </th>
  </tr>
</table>
</body>
</html>
"""

f.write(message)
f.close()

#Change path to reflect file location
webbrowser.open_new_tab(filename)