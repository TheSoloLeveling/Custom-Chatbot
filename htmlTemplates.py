js = '''
<script>
var i = 0;
var txt = document.getElementsByClassName('typewriter')[0].innerText;
var speed = 50; /* The speed/duration of the effect in milliseconds */


alert("test");
</script>
'''

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
.stTextInput>input {
    color: #ffffff; 
}
.stTextInput {
    color: #4F8BF9;
    position: fixed;
    bottom: 3rem;
    padding-right: 100px
}

.typewriter p {
  overflow: hidden; /* Ensures the content is not revealed until the animation */
  border-right: .05em solid orange; /* The typewriter cursor */
  white-space: pre-wrap; /* Keeps the content on a single line */
 /* Gives that scrolling effect as the typing happens */
  letter-spacing: .05em; /* Adjust as needed */
  font-size: 0.9em; /* Adjusts the size of the text */
  width: 100%; /* Sets the width of the container */
  text-overflow: ellipsis;
  animation: 
    typing 2.0s steps(40, end), /* Adjusts the speed of typing */
    blink-caret .75s step-end infinite;
}

/* The typing effect */
@keyframes typing {
  from { width: 0 }
  to { width: 100% }
}

/* The typewriter cursor effect */
@keyframes blink-caret {
  from, to { border-color: transparent }
  50% { border-color: orange; }
}
'''

bot_template = '''
<div class="chat-message bot typewriter">
    <div class="avatar">
        <img src="https://d2cbg94ubxgsnp.cloudfront.net/Pictures/2000x1125/9/9/3/512993_shutterstock_715962319converted_920340.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message"><p>{{MSG}}</p></div>
</div>
'''

user_template = '''
<div class="chat-message user ">
    <div class="avatar">
        <img src="https://t4.ftcdn.net/jpg/02/29/75/83/240_F_229758328_7x8jwCwjtBMmC6rgFzLFhZoEpLobB6L8.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''