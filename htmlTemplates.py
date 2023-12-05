
css = '''
<style>
.stTextInput>input {
    color: #ffffff; 
}
.stTextInput {
    color: #4F8BF9;
    position: fixed;
    bottom: 3rem;
    padding-right: 100px
}

'''

bot_template = '''
<div class="chat-message bot typewriter" id="typewriter">
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