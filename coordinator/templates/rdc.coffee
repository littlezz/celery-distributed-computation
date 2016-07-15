@tsss = () ->
  $('#p1').text('???')


# websocket
$ ->
  h = window.location.host
  ws = new WebSocket("ws://#{h}/ws_node_status")
  ws.onmessage = (e) ->
    msg = JSON.parse(e.data)
    $('#ws').text(msg.localhost)
    console.log msg
    ws.send('alive')


$ ->
  $('.chart').easyPieChart({
        animate: 2000,
        size:200,
    });
  x = () ->
        $('.chart').data('easyPieChart').update(40)
  setTimeout(x, 1000)

