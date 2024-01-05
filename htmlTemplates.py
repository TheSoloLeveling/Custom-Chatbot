
css = '''
<style>


.main {
    overflow: auto;
}


.stTextInput>input {
    color: #ffffff; 
}
.stTextInput {
    color: #4F8BF9;
    position: fixed;
    bottom: 3rem;  
}

.primaryButton {
    color: #4F8BF9;
    position: fixed;
    bottom: 3rem;
    border-radius: 50%;
    width: 40px;
    height: 40px;
}

.primaryButton[data-tooltip]:hover:after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: #333;
    color: #fff;
    padding: .5em;
    border-radius: .3em;
    font-size: 1.em; /* Adjust this to make the text larger or smaller */
    white-space: nowrap;
}

[data-testid="stSidebar"] .stButton {

    color: #4F8BF9;
    position: fixed;
    bottom: 1rem;

}

[data-testid='stFileUploader'] {
    color: #4F8BF9;
    position: fixed;
    bottom: 3rem; 
}



.appview-container {
            background: #262626
        }
        
'''
#262626
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