// Step 1: Set up the file and imports
// File: app/page.js
'use client'
import { Box, Button, Stack, TextField } from '@mui/material'
import { useState } from 'react'

// Step 2: Create the main component
export default function Home() {
  // Step 3: Set up state
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: `Hi! I'm the Rate My Professor support assistant. How can I help you today?`,
    },
  ])
  const [message, setMessage] = useState('')

  // Step 4: Implement the sendMessage function
  const sendMessage = async () => {
    setMessage('')
    setMessages((messages) => [
      ...messages,
      { role: 'user', content: message },
      { role: 'assistant', content: '' },
    ])

    const response = fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify([...messages, { role: 'user', content: message }]),
    }).then(async (res) => {
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let result = ''

      return reader.read().then(function processText({ done, value }) {
        if (done) {
          return result
        }
        const text = decoder.decode(value || new Uint8Array(), { stream: true })
        setMessages((messages) => {
          let lastMessage = messages[messages.length - 1]
          let otherMessages = messages.slice(0, messages.length - 1)
          return [
            ...otherMessages,
            { ...lastMessage, content: lastMessage.content + text },
          ]
        })
        return reader.read().then(processText)
      })
    })
  }

  // Step 5: Create the UI layout
  return (
    <Box
      width="100vw"
      height="100vh"
      display="flex"
      flexDirection="column"
      justifyContent="center"
      alignItems="center"
    >
      <Stack
        direction={'column'}
        width="500px"
        height="700px"
        border="1px solid black"
        p={2}
        spacing={3}
      >
        <Stack
          direction={'column'}
          spacing={2}
          flexGrow={1}
          overflow="auto"
          maxHeight="100%"
        >
          {messages.map((message, index) => (
            <Box
              key={index}
              display="flex"
              justifyContent={
                message.role === 'assistant' ? 'flex-start' : 'flex-end'
              }
            >
              <Box
                bgcolor={
                  message.role === 'assistant'
                    ? 'primary.main'
                    : 'secondary.main'
                }
                color="white"
                borderRadius={16}
                p={3}
              >
                {message.content}
              </Box>
            </Box>
          ))}
        </Stack>
        <Stack direction={'row'} spacing={2}>
  <TextField 
    label="Message" 
    fullWidth 
    value={message} 
    onChange={(e) => setMessage(e.target.value)}
    sx={{
      '& .MuiOutlinedInput-root': {
        '& fieldset': {
          borderColor: 'primary.main', // Match the button color
        },
        '&:hover fieldset': {
          borderColor: 'primary.main', // Match the button color on hover
        },
        '&.Mui-focused fieldset': {
          borderColor: 'primary.main', // Match the button color when focused
        },
      },
      '& .MuiInputBase-input': {
        color: 'white', // Set input text color to white
      },
      '& .MuiInputLabel-root': {
        color: 'white', // Set label color to white (optional)
      },
    }}
  />
  <Button variant="contained" onClick={sendMessage}>
    Send
  </Button>
</Stack>
      </Stack>
    </Box>
  )
}
