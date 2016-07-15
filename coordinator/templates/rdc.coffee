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