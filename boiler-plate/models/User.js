const mongoose = require('require');
const { mongo } = require('mongoose');


const userSchema = mongoose.Schema ({
    name: {
        type: String,
        maxlength: 50
    },
    email: {
        type: String,
        trim: true, //트림은 빈칸을 제거해 준다,무시한다
        unique: 1 //중복이메일 제거
    },
    password: {
        type: String,
        maxlength: 50
    },
    role: { //역할
        type: Number,
        default: 0
    },
    image: String,
    token: {
        type: String
    },
    tokenExp: { //토큰 유효기간
        type: Number
    }
});
// 모델의 이름 , 스키마
const User = mongoose.model('User', userSchema)

// 다른 곳에서 이 모델을 사용할 수 있게 해준다.
module.exports = { User }
