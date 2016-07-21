@tsss = () ->
  $('#p1').text('???')


node_dowm = (ip) ->
  $("#" + ip).transition('fade up')

node_up = (ip) ->
  $("#" + ip).transition('fade up')


# websocket
$ ->
  h = window.location.host
  ws = new WebSocket("ws://#{h}/ws_node_status")
  ws.onmessage = (e) ->
    msg = JSON.parse(e.data)
    console.log msg
    nodes_ip = Object.keys(msg.node_status)
    for ip in nodes_ip
      do (ip) ->
        old = $("#" + ip).find('span').text()
        pert = msg.node_status[ip]
        if pert == "offline"
          if old isnt "offline"
            console.log old
            node_dowm(ip)
        else if old == 'offline'
          node_up(ip)

        $("#" + ip).data('easyPieChart').update(pert)
        $("#" + ip).find('span').text(pert)
        $('#' + ip)


    nn_status = msg.nn_status
    $('#speed').contents()[2].nodeValue = nn_status.speed
    $('#rss').contents()[2].nodeValue = nn_status.rss
    ws.send('alive')


$ ->
  for name in ['localhost', 'node1', 'node2']
    do (name) ->

      $('#'+name).easyPieChart({
            animate: 600,
            size:200,
            barColor:'#ffff00',
            lineWidth:10,

        });

#  x = () ->
#        $('.chart').data('easyPieChart').update(40)
#  setTimeout(x, 1000)

