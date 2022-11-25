import webbrowser
from visualswarm.contrib.puppetmaster import ALL_HOSTS, WEBCAM_HOSTS

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
    document.getElementById('image_r1').src="http://{ALL_HOSTS["Robot1"]}:8000/"+ Date.now() +"stream.mjpg";
    document.getElementById('image_r2').src="http://{ALL_HOSTS["Robot2"]}:8000/"+ Date.now() +"stream.mjpg";
    document.getElementById('image_r3').src="http://{ALL_HOSTS["Robot3"]}:8000/"+ Date.now() +"stream.mjpg";
    document.getElementById('image_r4').src="http://{ALL_HOSTS["Robot4"]}:8000/"+ Date.now() +"stream.mjpg";
    document.getElementById('image_r5').src="http://{ALL_HOSTS["Robot5"]}:8000/"+ Date.now() +"stream.mjpg";
    document.getElementById('image_r6').src="http://{ALL_HOSTS["Robot6"]}:8000/"+ Date.now() +"stream.mjpg";
    document.getElementById('image_r7').src="http://{ALL_HOSTS["Robot7"]}:8000/"+ Date.now() +"stream.mjpg";
    document.getElementById('image_r8').src="http://{ALL_HOSTS["Robot8"]}:8000/"+ Date.now() +"stream.mjpg";
    document.getElementById('image_r9').src="http://{ALL_HOSTS["Robot9"]}:8000/"+ Date.now() +"stream.mjpg";
    document.getElementById('image_r10').src="http://{ALL_HOSTS["Robot10"]}:8000/"+ Date.now() +"stream.mjpg";
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
<th class="tg-0lax"><center>
<h1>Robot4</h1>
</center><center><img id="image_r4" width="100%" height="100%" /></center></th>
<th class="tg-0lax"><center>
<h1>Robot5</h1>
</center><center><img id="image_r5" width="100%" height="100%" /></center></th>
</tr>
</thead>
<tbody>
<tr>
<th class="tg-0lax"><center>
<h1>Robot6</h1>
</center><center><img id="image_r6" width="100%" height="100%" /></center></th>
<th class="tg-0lax"><center>
<h1>Robot7</h1>
</center><center><img id="image_r7" width="100%" height="100%" /></center></th>
<th class="tg-0lax"><center>
<h1>Robot8</h1>
</center><center><img id="image_r8" width="100%" height="100%" /></center></th>
<th class="tg-0lax"><center>
<h1>Robot9</h1>
</center><center><img id="image_r9" width="100%" height="100%" /></center></th>
<th class="tg-0lax"><center>
<h1>Robot10</h1>
</center><center><img id="image_r10" width="100%" height="100%" /></center></th>
</tr>
</html>
"""

f.write(message)
f.close()

#Change path to reflect file location
webbrowser.open_new_tab(filename)