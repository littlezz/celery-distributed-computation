@tsss = () ->
  $('#p1').text('???')


# websocket
$ ->
  h = window.location.host
  ws = new WebSocket("ws://#{h}/ws_node_status")
  ws.onmessage = (e) ->
    msg = JSON.parse(e.data)
#    pert = msg.node_status.localhost
#    $('.chart').data('easyPieChart').update(pert)
#    $('#percent1').text(pert)
    nodes_ip = Object.keys(msg.node_status)
    for ip in nodes_ip
      do (ip) ->
        pert = msg.node_status[ip]
        $("#" + ip).data('easyPieChart').update(pert)
        $("#" + ip).find('span').text(pert)


    
    console.log msg
    ws.send('alive')


$ ->
  for name in ['localhost', 'node1', 'node2']
    do (name) ->

      $('#'+name).easyPieChart({
            animate: 600,
            size:200,
            barColor:'#ef1e25',
            lineWidth:5,

        });

#  x = () ->
#        $('.chart').data('easyPieChart').update(40)
#  setTimeout(x, 1000)

