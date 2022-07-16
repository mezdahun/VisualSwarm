import webbrowser
from visualswarm.contrib.puppetmaster import HOSTS

filename = 'vision_summary.html'
f = open(filename, 'w')

message = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<title>VSWRM - Vision Summary</title>
</head>
<body>


<table class="tg" height="100%" width="100%">
<thead>
  <tr height="50%">
    <th class="tg-0lax">
        <center><h1>Robot1</h1></center>
        <center><img src="http://{HOSTS["Robot1"]}:8000/stream.mjpg" width="100%" height="100%"></center>
    </th>
    <th class="tg-0lax">
            <center><h1>Robot2</h1></center>
            <center><img src="http://{HOSTS["Robot2"]}:8000/stream.mjpg" width="100%" height="100%"></center></th>
    <th class="tg-0lax">
            <center><h1>Robot3</h1></center>
            <center><img src="http://{HOSTS["Robot3"]}:8000/stream.mjpg" width="100%" height="100%"></center></th>
  </tr>
</thead>
<tbody>
  <tr height="50%">
    <td class="tg-0lax">
            <center><h1>Robot4</h1></center>
            <center><img src="http://{HOSTS["Robot4"]}:8000/stream.mjpg" width="100%" height="100%"></center>
    </td>
    <td class="tg-0lax">
            <center><h1>Robot5</h1></center>
            <center><img src="http://{HOSTS["Robot5"]}:8000/stream.mjpg" width="100%" height="100%"></center>
    </td>
    <td class="tg-0lax"></td>
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