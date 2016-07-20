@tsss = () ->
  $('#p1').text('???')


# websocket
$ ->
  h = window.location.host
  ws = new WebSocket("ws://#{h}/ws_node_status")
  ws.onmessage = (e) ->
    msg = JSON.parse(e.data)
    pert = msg.node_status.localhost
    $('.chart').data('easyPieChart').update(pert)
    $('#percent1').text(pert)
    console.log msg
    ws.send('alive')


$ ->
  $('.chart').easyPieChart({
        animate: 600,
        size:200,
        barColor:'#ef1e25',
        lineWidth:5,

    });
#  x = () ->
#        $('.chart').data('easyPieChart').update(40)
#  setTimeout(x, 1000)

