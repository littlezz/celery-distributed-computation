@tsss = () ->
  $('#p1').text('???')


# websocket
$ ->
  h = window.location.host
  ws = new WebSocket("ws://#{h}/ws_node_status")
  ws.onmessage = (e) ->
    msg = JSON.parse(e.data)
    $('.chart').data('easyPieChart').update(msg.localhost)
    $('#percent1').text(msg.localhost)
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

